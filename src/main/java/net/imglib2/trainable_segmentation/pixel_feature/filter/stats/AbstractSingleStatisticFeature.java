package net.imglib2.trainable_segmentation.pixel_feature.filter.stats;

import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.trainable_segmentation.gpu.GpuFeatureInput;
import net.imglib2.trainable_segmentation.gpu.api.GpuApi;
import net.imglib2.trainable_segmentation.gpu.api.GpuView;
import net.imglib2.trainable_segmentation.pixel_feature.filter.AbstractFeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureInput;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import org.scijava.plugin.Parameter;
import net.imglib2.converter.RealTypeConverters;

import java.util.Collections;
import java.util.List;

abstract public class AbstractSingleStatisticFeature extends AbstractFeatureOp {
	@Parameter
	private double radius = 1;

	@Override
	public int count() {
		return 1;
	}

	@Override
	public List<String> attributeLabels() {
		return Collections.singletonList(filterName() + " filter radius=" + radius);
	}

	protected abstract String filterName();

	@Override
	public void apply(FeatureInput input, List<RandomAccessibleInterval<FloatType>> output) {
		int[] windowSize = getWindowSize();
		apply(windowSize, input.original(), output.get(0));
	}

	protected abstract void apply(int[] windowSize, RandomAccessible<FloatType> input, RandomAccessibleInterval<FloatType> output);

	@Override
	public void prefetch(GpuFeatureInput input) {
		input.prefetchOriginal(requiredSourceInterval(input.targetInterval()));
	}

	@Override
	public void apply(GpuFeatureInput input, List<GpuView> output) {
		try (GpuApi gpu = input.gpuApi().subScope()) {
			GpuView original = input.original(requiredSourceInterval(input.targetInterval()));
			int[] windowSize = getWindowSize();
			apply(gpu, windowSize, original, output.get(0));
		}
	}

	protected abstract void apply(GpuApi gpu, int[] windowSize, GpuView input, GpuView output);

	private int[] getWindowSize() {
		return globalSettings().pixelSize().stream()
				.mapToInt(pixelSize -> 1 + 2 * Math.max(0, (int) (radius / pixelSize)))
				.toArray();
	}

	private FinalInterval requiredSourceInterval(Interval targetInterval) {
		long[] borderSize = globalSettings().pixelSize().stream()
				.mapToLong(pixelSize -> Math.max(0, (long) (radius / pixelSize)))
				.toArray();
		return Intervals.expand(targetInterval, borderSize);
	}
}
