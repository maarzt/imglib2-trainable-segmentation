package net.imglib2.trainable_segmentation.pixel_feature.filter.stats;

import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.trainable_segmentation.gpu.algorithms.GpuNeighborhoodOperations;
import net.imglib2.trainable_segmentation.gpu.api.GpuApi;
import net.imglib2.trainable_segmentation.gpu.api.GpuImage;
import net.imglib2.trainable_segmentation.gpu.api.GpuPixelWiseOperation;
import net.imglib2.trainable_segmentation.gpu.api.GpuView;
import net.imglib2.trainable_segmentation.gpu.api.GpuViews;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureOp;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.scijava.plugin.Plugin;
import net.imglib2.algorithm.convolution.Convolution;
import net.imglib2.converter.RealTypeConverters;
import net.imglib2.loops.LoopBuilder;

@Plugin(type = FeatureOp.class, label = "variance filter")
public class SingleVarianceFeature extends AbstractSingleStatisticFeature {

	@Override
	protected String filterName() {
		return "variance";
	}

	@Override
	protected void apply(int[] windowSize, RandomAccessible<FloatType> input, RandomAccessibleInterval<FloatType> output) {
		long n = Intervals.numElements(windowSize);
		if (n <= 1) {
			LoopBuilder.setImages(output).forEachPixel(FloatType::setZero);
			return;
		}
		RandomAccessibleInterval<FloatType> mean = Views.translate(ArrayImgs.floats(Intervals.dimensionsAsLongArray(output)), Intervals.minAsLongArray(output));
		SumFilter.convolution(windowSize).process(input, mean);
		double factor = 1.0 / Intervals.numElements(windowSize);
		LoopBuilder.setImages(mean).forEachPixel(pixel -> pixel.mul(factor));
		RandomAccessible<FloatType> squared = Converters.convert(input, (i, o) -> o.set(square(i.getRealFloat())), new FloatType());
		Convolution<RealType<?>> sumFilter = SumFilter.convolution(windowSize);
		sumFilter.process(squared, output);
		float a = 1.0f / (n - 1);
		float b = (float) n / (n - 1);
		LoopBuilder.setImages(output, mean).forEachPixel((o, m) -> o.setReal(o.getRealFloat() * a -
				square(m.getRealFloat()) * b));
	}

	private float square(float x) {
		return x * x;
	}

	@Override
	protected void apply(GpuApi gpu, int[] windowSize, GpuView input, GpuView output) {
		long[] dimensions = Intervals.dimensionsAsLongArray(output.dimensions());
		GpuImage mean = gpu.create(dimensions, NativeTypeEnum.Float);
		GpuImage meanOfSquared = gpu.create(dimensions, NativeTypeEnum.Float);
		GpuImage squared = gpu.create(Intervals.dimensionsAsLongArray(input.dimensions()), NativeTypeEnum.Float);
		GpuNeighborhoodOperations.mean(gpu, windowSize).apply(input, GpuViews.wrap(mean));
		long n = Intervals.numElements(windowSize);
		if (n <= 1)
			GpuPixelWiseOperation.gpu(gpu).addOutput("variance", output).forEachPixel("variance = 0");
		else {
			square(gpu, input, squared);
			GpuNeighborhoodOperations.mean(gpu, windowSize).apply(GpuViews.wrap(squared), GpuViews.wrap(
					meanOfSquared));
			GpuPixelWiseOperation.gpu(gpu)
					.addInput("mean", mean)
					.addInput("mean_of_squared", meanOfSquared)
					.addInput("factor", (float) n / (n - 1))
					.addOutput("variance", output)
					.forEachPixel("variance = (mean_of_squared - mean * mean) * factor");
		}
	}

	private void square(GpuApi gpu, GpuView inputBuffer, GpuImage tmp2) {
		GpuPixelWiseOperation.gpu(gpu).addInput("a", inputBuffer).addOutput("b", tmp2)
				.forEachPixel("b = a * a");
	}

}
