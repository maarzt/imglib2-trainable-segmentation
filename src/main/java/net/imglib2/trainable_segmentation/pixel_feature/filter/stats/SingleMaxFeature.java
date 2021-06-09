package net.imglib2.trainable_segmentation.pixel_feature.filter.stats;

import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.trainable_segmentation.gpu.algorithms.GpuNeighborhoodOperations;
import net.imglib2.trainable_segmentation.gpu.api.GpuApi;
import net.imglib2.trainable_segmentation.gpu.api.GpuView;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureOp;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import org.scijava.plugin.Plugin;
import net.imglib2.converter.RealTypeConverters;

@Plugin(type = FeatureOp.class, label = "max filter")
public class SingleMaxFeature extends AbstractSingleStatisticFeature {

	@Override
	protected String filterName() {
		return "max";
	}

	@Override
	protected void apply(int[] windowSize, RandomAccessible<FloatType> input, RandomAccessibleInterval<FloatType> output) {
		if(Intervals.numElements(windowSize) <= 1)
			RealTypeConverters.copyFromTo(input, output);
		else
			MinMaxFilter.maxFilter(windowSize).process(input, output);
	}

	@Override
	protected void apply(GpuApi gpu, int[] windowSize, GpuView input, GpuView output) {
		GpuNeighborhoodOperations.max(gpu, windowSize).apply(input, output);
	}

}
