
package net.imglib2.trainable_segmention.pixel_feature.filter.hessian;

import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.trainable_segmentation.RevampUtils;
import net.imglib2.trainable_segmentation.pixel_feature.filter.AbstractFeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureInput;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureOp;
import net.imglib2.algorithm.gradient.PartialDerivative;
import net.imglib2.trainable_segmentation.pixel_feature.filter.hessian.EigenValuesSymmetric3D;
import net.imglib2.trainable_segmentation.pixel_feature.settings.GlobalSettings;
import net.imglib2.trainable_segmentation.utils.views.FastViews;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import net.imglib2.view.composite.Composite;
import net.imglib2.view.composite.RealComposite;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import net.imglib2.loops.LoopBuilder;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author Matthias Arzt
 */
@Deprecated
@Plugin(type = FeatureOp.class, label = "Hessian")
public class SingleHessian3DFeature extends AbstractFeatureOp {

	@Parameter
	double sigma = 4.0;

	@Parameter
	boolean absoluteValues = true;

	@Override
	public int count() {
		return 3;
	}

	@Override
	public List<String> attributeLabels() {
		return Stream.of("largest", "middle", "smallest").map(x -> "Hessian_" + x + "_" + sigma + "_" +
			absoluteValues)
			.collect(Collectors.toList());
	}

	@Override
	public void apply(FeatureInput input, List<RandomAccessibleInterval<FloatType>> output) {
		calculateHessianOnChannel(input.original(), output, sigma);
	}

	@Override
	public boolean checkGlobalSettings(GlobalSettings globals) {
		return globals.numDimensions() == 3;
	}

	private void calculateHessianOnChannel(RandomAccessible<FloatType> image,
		List<RandomAccessibleInterval<FloatType>> output, double sigma)
	{
		double[] sigmas = { 0.4 * sigma, 0.4 * sigma, 0.4 * sigma };

		Interval secondDerivativeInterval = output.get(0);
		Interval firstDerivativeInterval = Intervals.expand(secondDerivativeInterval, 1);
		Interval blurredInterval = Intervals.expand(firstDerivativeInterval, 1);

		RandomAccessibleInterval<FloatType> blurred = RevampUtils.gauss(image, blurredInterval,
			sigmas);
		RandomAccessibleInterval<FloatType> dx = derive(blurred, firstDerivativeInterval, 0);
		RandomAccessibleInterval<FloatType> dy = derive(blurred, firstDerivativeInterval, 1);
		RandomAccessibleInterval<FloatType> dz = derive(blurred, firstDerivativeInterval, 2);
		RandomAccessibleInterval<Composite<FloatType>> secondDerivatives =
			calculateSecondDerivatives(secondDerivativeInterval, dx, dy, dz);

		RandomAccessibleInterval<Composite<FloatType>> eigenValues = RevampUtils.vectorizeStack(output);
		EigenValuesSymmetric3D<FloatType, FloatType> ev = new EigenValuesSymmetric3D<>();
		if (absoluteValues)
			LoopBuilder.setImages(secondDerivatives, eigenValues).forEachPixel(
				ev::computeMagnitudeOfEigenvalues);
		else
			LoopBuilder.setImages(secondDerivatives, eigenValues).forEachPixel(ev::compute);
	}

	private RandomAccessibleInterval<Composite<FloatType>> calculateSecondDerivatives(
		Interval secondDerivativeInterval,
		RandomAccessibleInterval<FloatType> dx,
		RandomAccessibleInterval<FloatType> dy,
		RandomAccessibleInterval<FloatType> dz)
	{
		RandomAccessibleInterval<FloatType> secondDerivatives = RevampUtils.createImage(
			RevampUtils.appendDimensionToInterval(secondDerivativeInterval, 0, 5), new FloatType());
		List<RandomAccessibleInterval<FloatType>> slices = RevampUtils.slices(secondDerivatives);
		PartialDerivative.gradientCentralDifference(dx, slices.get(0), 0);
		PartialDerivative.gradientCentralDifference(dx, slices.get(1), 1);
		PartialDerivative.gradientCentralDifference(dx, slices.get(2), 2);
		PartialDerivative.gradientCentralDifference(dy, slices.get(3), 1);
		PartialDerivative.gradientCentralDifference(dy, slices.get(4), 2);
		PartialDerivative.gradientCentralDifference(dz, slices.get(5), 2);
		return FastViews.collapse(secondDerivatives);
	}

	// -- Helper methods --

	private RandomAccessibleInterval<FloatType> derive(RandomAccessible<FloatType> source,
		Interval interval, int dimension)
	{
		RandomAccessibleInterval<FloatType> target = RevampUtils.createImage(interval, new FloatType());
		PartialDerivative.gradientCentralDifference(source, target, dimension);
		return target;
	}
}
