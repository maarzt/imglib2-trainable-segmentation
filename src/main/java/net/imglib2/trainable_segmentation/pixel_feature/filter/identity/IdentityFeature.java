
package net.imglib2.trainable_segmentation.pixel_feature.filter.identity;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.trainable_segmentation.gpu.api.GpuCopy;
import net.imglib2.trainable_segmentation.gpu.GpuFeatureInput;
import net.imglib2.trainable_segmentation.gpu.api.GpuView;
import net.imglib2.trainable_segmentation.pixel_feature.filter.AbstractFeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureInput;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureOp;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.scijava.plugin.Plugin;
import net.imglib2.loops.LoopBuilder;

import java.util.Collections;
import java.util.List;

/**
 * @author Matthias Arzt
 */

@Plugin(type = FeatureOp.class, label = "original image")
public class IdentityFeature extends AbstractFeatureOp {

	@Override
	public int count() {
		return 1;
	}

	@Override
	public void apply(FeatureInput input, List<RandomAccessibleInterval<FloatType>> output) {
		RandomAccessibleInterval<FloatType> in = Views.interval(input.original(), input
			.targetInterval());
		LoopBuilder.setImages(in, output.get(0)).forEachPixel((i, o) -> o.set(i));
	}

	@Override
	public void prefetch(GpuFeatureInput input) {
		input.prefetchOriginal(input.targetInterval());
	}

	@Override
	public void apply(GpuFeatureInput input, List<GpuView> output) {
		GpuView in = input.original(input.targetInterval());
		GpuCopy.copyFromTo(input.gpuApi(), in, output.get(0));
	}

	@Override
	public List<String> attributeLabels() {
		return Collections.singletonList("original");
	}
}
