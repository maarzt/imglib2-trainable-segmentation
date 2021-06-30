
package net.imglib2.trainable_segmentation.pixel_feature.calculator;

import net.imglib2.trainable_segmentation.gpu.api.GpuPixelWiseOperation;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.trainable_segmentation.Utils;
import net.imglib2.trainable_segmentation.gpu.GpuFeatureInput;
import net.imglib2.trainable_segmentation.gpu.api.GpuView;
import net.imglib2.trainable_segmentation.pixel_feature.filter.AbstractFeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureInput;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.settings.ChannelSetting;
import net.imglib2.trainable_segmentation.pixel_feature.settings.FeatureSetting;
import net.imglib2.trainable_segmentation.utils.CpuGpuRunner;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.scijava.plugin.Parameter;
import net.imglib2.loops.LoopBuilder;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

@RunWith(CpuGpuRunner.class)
public class FeatureCalculatorTest {

	public FeatureCalculatorTest(boolean useGpu) {
		this.useGpu = useGpu;
	}

	private final boolean useGpu;

	private final FeatureSetting add_42 = new FeatureSetting(AddValue.class, "value", 42);
	private final FeatureSetting add_12 = new FeatureSetting(AddValue.class, "value", 12);

	@Test
	public void test1() {
		FeatureCalculator calculator = FeatureCalculator.default2d()
			.addFeatures(add_42, add_12)
			.build();
		calculator.setUseGpu(useGpu);
		Img<FloatType> input = ArrayImgs.floats(new float[] { 2 }, 1, 1);
		RandomAccessibleInterval<FloatType> out = calculator.apply(input);
		Utils.assertImagesEqual(ArrayImgs.floats(new float[] { 44, 14 }, 1, 1, 2), out);
	}

	@Test
	public void test2() {
		FeatureCalculator calculator = FeatureCalculator.default2d()
			.channels(ChannelSetting.multiple(2))
			.sigmas(1.0)
			.addFeatures(add_42, add_12)
			.build();
		calculator.setUseGpu(useGpu);
		Img<FloatType> input = ArrayImgs.floats(new float[] { 2, 3 }, 1, 1, 2);
		RandomAccessibleInterval<FloatType> out = calculator.apply(input);
		assertEquals(Arrays.asList("channel1_add_value_42.0", "channel2_add_value_42.0",
			"channel1_add_value_12.0", "channel2_add_value_12.0"),
			calculator.attributeLabels());
		Utils.assertImagesEqual(ArrayImgs.floats(new float[] { 44, 45, 14, 15 }, 1, 1, 4), out);
	}

	public static class AddValue extends AbstractFeatureOp implements FeatureOp {

		@Parameter
		double value;

		@Override
		public int count() {
			return 1;
		}

		@Override
		public List<String> attributeLabels() {
			return Collections.singletonList("add_value_" + value);
		}

		@Override
		public void apply(FeatureInput input,
			List<RandomAccessibleInterval<FloatType>> output)
		{
			IntervalView<FloatType> inputInterval = Views.interval(input.original(), output.get(0));
			LoopBuilder.setImages(inputInterval, output.get(0)).multiThreaded().forEachPixel(
				(in, out) -> out.set(in.get() + (float) value));
		}

		@Override
		public void prefetch(GpuFeatureInput input) {
			input.prefetchOriginal(input.targetInterval());
		}

		@Override
		public void apply(GpuFeatureInput input, List<GpuView> output) {
			GpuView image = input.original(input.targetInterval());
			GpuPixelWiseOperation.gpu(input.gpuApi())
				.addInput("a", image)
				.addInput("b", (float) value)
				.addOutput("c", output.get(0)).forEachPixel("c = a + b");
		}
	}
}
