
package net.imglib2.trainable_segmention.pixel_feature.filter.gradient;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.trainable_segmentation.RevampUtils;
import net.imglib2.trainable_segmentation.pixel_feature.filter.AbstractFeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureInput;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author Matthias Arzt
 */
@Deprecated
@Plugin(type = FeatureOp.class, label = "Gradient")
public class SingleGradientFeature extends AbstractFeatureOp {

	@Parameter
	private double sigma = 1.0;

	@Override
	public int count() {
		return 1;
	}

	@Override
	public List<String> attributeLabels() {
		return Collections.singletonList("Gradient_filter_" + sigma);
	}

	@Override
	public void apply(FeatureInput input, List<RandomAccessibleInterval<FloatType>> output) {
		final int n = globalSettings().numDimensions();
		List<RandomAccessibleInterval<DoubleType>> derivatives = IntStream
			.range(0, n)
			.mapToObj(d -> derive(input, d)).collect(Collectors.toList());
		LoopBuilder.setImages(RevampUtils.vectorizeStack(derivatives), output.get(0)).forEachPixel((in,
			out) -> {
			double sum = 0;
			for (int i = 0; i < n; i++) {
				double x = in.get(i).getRealDouble();
				sum += x * x;
			}
			out.setReal(Math.sqrt(sum));
		});
	}

	private RandomAccessibleInterval<DoubleType> derive(FeatureInput input, int d) {
		int[] orders = IntStream.range(0, globalSettings().numDimensions())
			.map(i -> i == d ? 1 : 0).toArray();
		return input.derivedGauss(sigma * 0.4, orders);
	}
}
