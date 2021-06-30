
package net.imglib2.trainable_segmentation.pixel_feature.filter.hessian;

import net.imglib2.trainable_segmentation.gpu.algorithms.GpuEigenvalues;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.trainable_segmentation.gpu.GpuFeatureInput;
import net.imglib2.trainable_segmentation.gpu.api.GpuView;
import net.imglib2.trainable_segmentation.pixel_feature.filter.AbstractFeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureInput;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.settings.GlobalSettings;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import net.imglib2.loops.LoopBuilder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Plugin(type = FeatureOp.class, label = "hessian eigenvalues")
public class SingleHessianEigenvaluesFeature extends AbstractFeatureOp {

	@Parameter
	private double sigma = 1.0;

	@Override
	public int count() {
		return globalSettings().numDimensions();
	}

	@Override
	public void apply(FeatureInput input, List<RandomAccessibleInterval<FloatType>> output) {
		if (globalSettings().numDimensions() == 2)
			apply2d(input, output);
		else if (globalSettings().numDimensions() == 3)
			apply3d(input, output);
		else throw new AssertionError();
	}

	@Override
	public List<String> attributeLabels() {
		List<String> prefix = globalSettings().numDimensions() == 2 ? Arrays.asList("largest",
			"smallest") : Arrays.asList("largest", "middle", "smallest");
		return prefix.stream().map(s -> "hessian - " + s + " eigenvalue sigma=" + sigma)
			.collect(Collectors.toList());
	}

	@Override
	public boolean checkGlobalSettings(GlobalSettings globals) {
		return globals.numDimensions() == 2 || globals.numDimensions() == 3;
	}

	private void apply2d(FeatureInput input, List<RandomAccessibleInterval<FloatType>> output) {
		RandomAccessibleInterval<DoubleType> dxx = input.derivedGauss(sigma, 2, 0);
		RandomAccessibleInterval<DoubleType> dxy = input.derivedGauss(sigma, 1, 1);
		RandomAccessibleInterval<DoubleType> dyy = input.derivedGauss(sigma, 0, 2);
		RandomAccessibleInterval<FloatType> larger = output.get(0);
		RandomAccessibleInterval<FloatType> smaller = output.get(1);
		LoopBuilder.setImages(dxx, dxy, dyy, larger, smaller).multiThreaded().forEachPixel(
			(s_xx, s_yx, s_yy, l, s) -> calculateHessianPerPixel(s_xx.getRealDouble(), s_yx
				.getRealDouble(), s_yy.getRealDouble(), l, s));
	}

	private static void calculateHessianPerPixel(
		double s_xx, double s_xy, double s_yy, FloatType largerEigenvalue, FloatType smallerEigenvalue)
	{
		final double trace = s_xx + s_yy;
		float l = (float) (trace / 2.0 + Math.sqrt(4 * s_xy * s_xy + (s_xx - s_yy) * (s_xx - s_yy)) /
			2.0);
		largerEigenvalue.set(l);
		float s = (float) (trace / 2.0 - Math.sqrt(4 * s_xy * s_xy + (s_xx - s_yy) * (s_xx - s_yy)) /
			2.0);
		smallerEigenvalue.set(s);
	}

	private void apply3d(FeatureInput input, List<RandomAccessibleInterval<FloatType>> output) {
		RandomAccessibleInterval<DoubleType> a11 = input.derivedGauss(sigma, 2, 0, 0);
		RandomAccessibleInterval<DoubleType> a12 = input.derivedGauss(sigma, 1, 1, 0);
		RandomAccessibleInterval<DoubleType> a13 = input.derivedGauss(sigma, 1, 0, 1);
		RandomAccessibleInterval<DoubleType> a22 = input.derivedGauss(sigma, 0, 2, 0);
		RandomAccessibleInterval<DoubleType> a23 = input.derivedGauss(sigma, 0, 1, 1);
		RandomAccessibleInterval<DoubleType> a33 = input.derivedGauss(sigma, 0, 0, 2);
		EigenValuesSymmetric3D.calc(a11, a12, a13, a22, a23, a33, output);
	}

	@Override
	public void prefetch(GpuFeatureInput input) {
		final Interval interval = input.targetInterval();
		final int n = interval.numDimensions();
		for (int d1 = 0; d1 < n; d1++)
			for (int d2 = d1; d2 < n; d2++)
				input.prefetchSecondDerivative(sigma, d1, d2, interval);
	}

	@Override
	public void apply(GpuFeatureInput input, List<GpuView> output) {
		final Interval interval = input.targetInterval();
		final int n = interval.numDimensions();
		final List<GpuView> derivatives = new ArrayList<>();
		for (int d1 = 0; d1 < n; d1++)
			for (int d2 = d1; d2 < n; d2++)
				derivatives.add(input.secondDerivative(sigma, d1, d2, interval));
		List<GpuView> eigenvalues = new ArrayList<>(output);
		GpuEigenvalues.symmetric(input.gpuApi(), derivatives, eigenvalues);
	}
}
