
package net.imglib2.trainable_segmentation.pixel_feature.filter;

import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.Localizable;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.test.ImgLib2Assert;
import net.imglib2.algorithm.convolution.kernel.Kernel1D;
import net.imglib2.algorithm.convolution.kernel.SeparableKernelConvolution;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.converter.Converters;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.trainable_segmentation.Utils;
import net.imglib2.trainable_segmentation.pixel_feature.filter.gradient.DerivedNormalDistribution;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.ConstantUtils;
import net.imglib2.util.Intervals;
import net.imglib2.util.Localizables;
import net.imglib2.view.Views;
import org.junit.Test;

import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;

public class FeatureInputTest {

	private RandomAccessibleInterval<FloatType> image = initImage();

	private Interval target = Intervals.createMinSize(10, 10, 10, 10, 10, 10);

	private FeatureInput cache = new FeatureInput(image, target, new double[] { 1, 1, 2 });

	private RandomAccessibleInterval<FloatType> initImage() {
		RandomAccessibleInterval<FloatType> result = ArrayImgs.floats(30, 30, 30);
		Views.interval(result, Intervals.createMinSize(15, 15, 15, 15, 15, 15)).forEach(
			RealType::setOne);
		return result;
	}

	@Test
	public void testSource() {
		assertSame(image, cache.original());
	}

	@Test
	public void testGauss() {
		RandomAccessibleInterval<? extends RealType<?>> gauss = cache.gauss(2.0);
		RandomAccessibleInterval<DoubleType> expected = create(target);
		Gauss3.gauss(new double[] { 2.0, 2.0, 1.0 }, image, expected);
		Utils.assertImagesEqual(expected, (RandomAccessibleInterval) gauss);
	}

	@Test
	public void testGaussCache() {
		// assertSame(cache.gauss(1.0), cache.gauss(1.0));
		assertNotSame(cache.gauss(2.0), cache.gauss(1.0));
	}

	@Test
	public void testDerivedGauss() {
		RandomAccessibleInterval<? extends RealType<?>> result = cache.derivedGauss(2.0, 0, 1, 0);
		RandomAccessibleInterval<DoubleType> expected = deriveY();
		ImgLib2Assert.assertImageEqualsRealType(expected, (RandomAccessibleInterval) result, 0.01);
	}

	@Test
	public void testDerivedGaussCache() {
		assertSame(cache.derivedGauss(2.0, 1, 0, 0), cache.derivedGauss(2.0, 1, 0, 0));
		assertNotSame(cache.derivedGauss(2.0, 1, 0, 0), cache.derivedGauss(2.0, 0, 0, 0));
		assertNotSame(cache.derivedGauss(3.0, 1, 0, 0), cache.derivedGauss(2.0, 1, 0, 0));
		assertNotSame(cache.derivedGauss(2.0, 1, 0, 0), cache.derivedGauss(2.0, 1, 0));
	}

	private RandomAccessibleInterval<DoubleType> deriveY() {
		RandomAccessibleInterval<DoubleType> expected = create(target);
		Kernel1D[] kernels = {
			DerivedNormalDistribution.derivedGaussKernel(2.0, 0),
			DerivedNormalDistribution.derivedGaussKernel(2.0, 1),
			DerivedNormalDistribution.derivedGaussKernel(1.0, 0),
		};
		SeparableKernelConvolution.convolve(kernels, image, expected);
		return expected;
	}

	private RandomAccessibleInterval<DoubleType> create(Interval target) {
		Img<DoubleType> result = ArrayImgs.doubles(Intervals.dimensionsAsLongArray(target));
		return Views.translate(result, Intervals.minAsLongArray(target));
	}

	@Test
	public void testCorrectDerivatives() {
		RandomAccessible<Localizable> position = Localizables.randomAccessible(3);
		RandomAccessible<FloatType> image = Converters.convert(position, (pos, pixel) -> {
			double x = pos.getDoublePosition(0);
			double y = pos.getDoublePosition(1);
			double z = pos.getDoublePosition(2) * 3;
			pixel.setReal(1 * x + 2 * y + 3 * z);
		}, new FloatType());
		double[] pixelSize = { 1, 1, 3 };
		FeatureInput input = new FeatureInput(image, new FinalInterval(1, 1, 1), pixelSize);
		assertEquals(1.0, getValue(input.derivedGauss(1, 1, 0, 0)), 0.001);
		assertEquals(2.0, getValue(input.derivedGauss(1, 0, 1, 0)), 0.002);
		assertEquals(3.0, getValue(input.derivedGauss(8, 0, 0, 1)), 0.001);
	}

	@Test
	public void testKernel() {
		testDerivativeKernel(x -> 1, 0, 1.0);
		testDerivativeKernel(x -> 1, 1, 0.0);
		testDerivativeKernel(x -> 1, 2, 0.0);
		testDerivativeKernel(x -> x, 0, 0.0);
		testDerivativeKernel(x -> x, 1, 1.0);
		testDerivativeKernel(x -> x, 2, 0.0);
		testDerivativeKernel(x -> Math.pow(x, 2), 1, 0.0);
		testDerivativeKernel(x -> Math.pow(x, 2), 2, 2.0);
		testDerivativeKernel(x -> 5 * x * x + 7 * x + 12, 0, 12);
		testDerivativeKernel(x -> 5 * x * x + 7 * x + 12, 1, 7);
		testDerivativeKernel(x -> 5 * x * x + 7 * x + 12, 2, 2 * 5);
	}

	private void testDerivativeKernel(DoubleUnaryOperator input, int order, double expected) {
		double[] inputValues = IntStream.range(-2, 2).asDoubleStream().map(input).toArray();
		RandomAccessibleInterval<DoubleType> image = Views.translate(ArrayImgs.doubles(inputValues, 5),
			-2);
		Kernel1D kernel = FeatureInput.SIMPLE_KERNELS.get(order);
		Img<DoubleType> result = ArrayImgs.doubles(1);
		SeparableKernelConvolution.convolution(kernel).process(image, result);
		assertEquals(expected, result.firstElement().getRealDouble(), 0.01);
	}

	private double getValue(RandomAccessibleInterval<DoubleType> actual) {
		RandomAccess<DoubleType> ra = actual.randomAccess();
		ra.setPosition(new long[] { 0, 0, 0 });
		return ra.get().getRealDouble();
	}

	private RandomAccessibleInterval<DoubleType> constant(double v, Interval interval) {
		return Views.interval(ConstantUtils.constantRandomAccessible(new DoubleType(v), interval
			.numDimensions()), interval);
	}
}
