
package net.imglib2.trainable_segmention.pixel_feature.filter.lipschitz;

import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.trainable_segmentation.RevampUtils;
import net.imglib2.trainable_segmentation.pixel_feature.filter.AbstractFeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureInput;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.settings.GlobalSettings;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.Views;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import net.imglib2.loops.LoopBuilder;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * @author Matthias Arzt
 */
@Deprecated
@Plugin(type = FeatureOp.class, label = "Lipschitz")
public class SingleLipschitzFeature extends AbstractFeatureOp {

	@Parameter
	private double slope;

	@Parameter
	private long border;

	public SingleLipschitzFeature() {}

	public SingleLipschitzFeature(double slope, long border) {
		this.slope = slope;
		this.border = border;
	}

	@Override
	public int count() {
		return 1;
	}

	@Override
	public void apply(FeatureInput input, List<RandomAccessibleInterval<FloatType>> output) {
		apply(input, output.get(0));
	}

	@Override
	public boolean checkGlobalSettings(GlobalSettings globals) {
		return globals.numDimensions() == 2;
	}

	private void apply(FeatureInput in, RandomAccessibleInterval<FloatType> out) {
		RandomAccessible<FloatType> original = in.original();
		double[] pixelSize = globalSettings().pixelSizeAsDoubleArray();
		Interval expandedInterval = Intervals.expand(out, scaledBorder(pixelSize, out));
		RandomAccessibleInterval<FloatType> tmp = RevampUtils.createImage(expandedInterval,
			new FloatType());
		copy(original, tmp);
		ConeMorphology.performConeOperation(ConeMorphology.Operation.DILATION, tmp, scaledSlope(
			pixelSize));
		outEquals255PlusAMinusB(out, original, tmp); // out = 255 + in - tmp
	}

	private double[] scaledSlope(double[] pixelSizes) {
		return Arrays.stream(pixelSizes).map(pixelSize -> slope * pixelSize).toArray();
	}

	private long[] scaledBorder(double[] pixelSizes, RandomAccessibleInterval<FloatType> out) {
		return Arrays.stream(pixelSizes).mapToLong(pixelSize -> (long) Math.ceil(border / pixelSize))
			.toArray();
	}

	private <T extends Type<T>> void copy(RandomAccessible<T> in, RandomAccessibleInterval<T> out) {
		LoopBuilder.setImages(Views.interval(in, out), out).multiThreaded().forEachPixel((i, o) -> o
			.set(i));
	}

	private <T extends RealType<T>> void outEquals255PlusAMinusB(RandomAccessibleInterval<T> out,
		RandomAccessible<T> A, RandomAccessible<T> B)
	{
		T offset = Util.getTypeFromInterval(out).createVariable();
		offset.setReal(255);
		LoopBuilder.setImages(Views.interval(A, out), Views.interval(B, out), out).forEachPixel((a, b,
			o) -> {
			o.set(offset);
			o.sub(b);
			o.add(a);
		});
	}

	@Override
	public List<String> attributeLabels() {
		return Collections.singletonList("Lipschitz_true_true_" + slope);
	}

}
