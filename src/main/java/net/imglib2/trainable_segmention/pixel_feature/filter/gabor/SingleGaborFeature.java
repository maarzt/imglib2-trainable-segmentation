
package net.imglib2.trainable_segmention.pixel_feature.filter.gabor;

import net.imagej.ops.OpEnvironment;
import net.imglib2.*;
import net.imglib2.trainable_segmentation.RevampUtils;
import net.imglib2.algorithm.fft2.FFTConvolution;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.trainable_segmentation.pixel_feature.filter.AbstractFeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureInput;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.settings.GlobalSettings;
import net.imglib2.trainable_segmentation.utils.views.FastViews;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import net.imglib2.view.composite.Composite;
import net.imglib2.view.composite.GenericComposite;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import net.imglib2.loops.LoopBuilder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @deprecated
 * @author Matthias Arzt
 */
@Deprecated
@Plugin(type = FeatureOp.class, label = "Gabor")
public class SingleGaborFeature extends AbstractFeatureOp {

	@Parameter
	private double sigma;

	@Parameter
	private double gamma;

	@Parameter
	private double psi;

	@Parameter
	private double frequency;

	@Parameter
	private int nAngles;

	@Parameter
	private boolean legacyNormalize = false;

	private List<RandomAccessibleInterval<FloatType>> kernels;

	@Override
	public void initialize() {
		if (sigma == 0.0)
			throw new AssertionError("sigma must be non zero.");
		kernels = initGaborKernels(sigma, gamma, psi, frequency, nAngles);
	}

	@Override
	public int count() {
		return 2;
	}

	@Override
	public void apply(FeatureInput in, List<RandomAccessibleInterval<FloatType>> out) {
		gaborProcessChannel(kernels, in.original(), out.get(0), out.get(1));
	}

	@Override
	public List<String> attributeLabels() {
		String details = "_" + sigma + "_" + gamma + "_" + (int) (psi / (Math.PI / 4)) + "_" +
			frequency;
		return Arrays.asList("Gabor_1" + details, "Gabor_2" + details);
	}

	@Override
	public boolean checkGlobalSettings(GlobalSettings globals) {
		return globals.numDimensions() == 2;
	}

	private List<RandomAccessibleInterval<FloatType>> initGaborKernels(double sigma, double gamma,
		double psi,
		double frequency, int nAngles)
	{
		// Apply aspect ratio to the Gaussian curves
		final double sigma_x = sigma;
		final double sigma_y = sigma / gamma;

		// Decide size of the filters based on the sigma
		int largerSigma = (sigma_x > sigma_y) ? (int) sigma_x : (int) sigma_y;
		if (largerSigma < 1)
			largerSigma = 1;

		// Create set of filters
		final int filterSizeX = 6 * largerSigma + 1;
		final int filterSizeY = 6 * largerSigma + 1;

		final int middleX = Math.round(filterSizeX / 2);
		final int middleY = Math.round(filterSizeY / 2);

		List<RandomAccessibleInterval<FloatType>> kernels = new ArrayList<>();

		final double rotationAngle = Math.PI / nAngles;

		FinalInterval interval = new FinalInterval(new long[] { -middleX, -middleY }, new long[] {
			middleX, middleX });
		for (int i = 0; i < nAngles; i++) {
			final double theta = rotationAngle * i;
			RandomAccessibleInterval<FloatType> kernel = RevampUtils.createImage(interval,
				new FloatType());
			garborKernel(kernel, psi, frequency, sigma_x, sigma_y, theta);
			kernels.add(kernel);
		}
		return kernels;
	}

	private static void garborKernel(RandomAccessibleInterval<FloatType> kernel, double psi,
		double frequency,
		double sigma_x, double sigma_y, double theta)
	{
		Cursor<FloatType> cursor = Views.iterable(kernel).cursor();
		final double filterSizeX = kernel.max(0) - kernel.min(0) + 1;
		final double sigma_x2 = sigma_x * sigma_x;
		final double sigma_y2 = sigma_y * sigma_y;
		while (cursor.hasNext()) {
			cursor.next();
			double x = cursor.getDoublePosition(0);
			double y = cursor.getDoublePosition(1);
			final double xPrime = x * Math.cos(theta) + y * Math.sin(theta);
			final double yPrime = y * Math.cos(theta) - x * Math.sin(theta);
			final double a = 1.0 / (2 * Math.PI * sigma_x * sigma_y) * Math.exp(-0.5 * (xPrime * xPrime /
				sigma_x2 + yPrime * yPrime / sigma_y2));
			final double c = Math.cos(2 * Math.PI * (frequency * xPrime) / filterSizeX + psi);
			cursor.get().set((float) (a * c));
		}
	}

	private void gaborProcessChannel(List<RandomAccessibleInterval<FloatType>> kernels,
		RandomAccessible<FloatType> channel, RandomAccessibleInterval<FloatType> max,
		RandomAccessibleInterval<FloatType> min)
	{
		Interval interval = min;
		RandomAccessibleInterval<FloatType> stack = RevampUtils.createImage(RevampUtils
			.appendDimensionToInterval(interval, 0,
				kernels.size() - 1), new FloatType());
		// Apply kernels
		FFTConvolution<FloatType> fftConvolution = new FFTConvolution<>(channel, interval, kernels.get(
			0), (Interval) kernels.get(0), new ArrayImgFactory<>());
		fftConvolution.setKeepImgFFT(true);
		for (int i = 0; i < kernels.size(); i++) {
			RandomAccessibleInterval<FloatType> kernel = kernels.get(i);
			RandomAccessibleInterval<FloatType> slice = Views.hyperSlice(stack, 2, i);
			fftConvolution.setKernel(kernel);
			fftConvolution.setOutput(slice);
			fftConvolution.convolve();
			if (legacyNormalize)
				normalize(ops(), slice);
		}

		maxAndMinProjection(stack, max, min);
	}

	private static void maxAndMinProjection(RandomAccessibleInterval<FloatType> stack,
		RandomAccessibleInterval<FloatType> max, RandomAccessibleInterval<FloatType> min)
	{
		RandomAccessibleInterval<Composite<FloatType>> collapsed = FastViews.collapse(stack);
		long size = stack.max(2) - stack.min(2) + 1;
		LoopBuilder.setImages(collapsed, max).forEachPixel((in, out) -> out.set(max(in, size)));
		LoopBuilder.setImages(collapsed, min).forEachPixel((in, out) -> out.set(min(in, size)));
	}

	private static float max(Composite<FloatType> in, long size) {
		float max = Float.NEGATIVE_INFINITY;
		for (int i = 0; i < size; i++)
			max = Math.max(max, in.get(i).get());
		return max;
	}

	private static float min(Composite<FloatType> in, long size) {
		float min = Float.POSITIVE_INFINITY;
		for (int i = 0; i < size; i++)
			min = Math.min(min, in.get(i).get());
		return min;
	}

	static void normalize(OpEnvironment ops, RandomAccessibleInterval<FloatType> image2) {
		DoubleType mean = ops.stats().mean(Views.iterable(image2));
		DoubleType stdDev = ops.stats().stdDev(Views.iterable(image2));
		float mean2 = (float) mean.get();
		float invStdDev = (stdDev.get() == 0) ? 1 : (float) (1 / stdDev.get());
		Views.iterable(image2).forEach(value -> value.set((value.get() - mean2) * invStdDev));
	}
}
