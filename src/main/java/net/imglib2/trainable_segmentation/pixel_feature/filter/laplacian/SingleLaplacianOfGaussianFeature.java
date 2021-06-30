
package net.imglib2.trainable_segmentation.pixel_feature.filter.laplacian;

import net.imglib2.trainable_segmentation.gpu.api.GpuPixelWiseOperation;
import net.imglib2.trainable_segmentation.gpu.api.GpuApi;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.trainable_segmentation.RevampUtils;
import net.imglib2.trainable_segmentation.gpu.GpuFeatureInput;
import net.imglib2.trainable_segmentation.gpu.api.GpuView;
import net.imglib2.trainable_segmentation.pixel_feature.filter.AbstractFeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureInput;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureOp;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.composite.Composite;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import net.imglib2.loops.LoopBuilder;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@Plugin(type = FeatureOp.class, label = "laplacian of gaussian")
public class SingleLaplacianOfGaussianFeature extends AbstractFeatureOp {

	@Parameter
	private double sigma = 1.0;

	@Override
	public int count() {
		return 1;
	}

	@Override
	public List<String> attributeLabels() {
		return Collections.singletonList("laplacian of gaussian sigma=" + sigma);
	}

	@Override
	public void apply(FeatureInput input, List<RandomAccessibleInterval<FloatType>> output) {
		int n = globalSettings().numDimensions();
		switch (n) {
			case 2:
				apply2d(input, output);
				return;
			case 3:
				apply3d(input, output);
				return;
			default:
				throw new IllegalArgumentException("Expect 2d or 3d.");
		}
	}

	private void apply2d(FeatureInput input, List<RandomAccessibleInterval<FloatType>> output) {
		RandomAccessibleInterval<DoubleType> dx = input.derivedGauss(sigma, order(2, 0));
		RandomAccessibleInterval<DoubleType> dy = input.derivedGauss(sigma, order(2, 1));
		LoopBuilder.setImages(dx, dy, output.get(0)).multiThreaded().forEachPixel(
			(x, y, sum) -> sum.setReal(x.getRealDouble() + y.getRealDouble()));
	}

	private void apply3d(FeatureInput input, List<RandomAccessibleInterval<FloatType>> output) {
		RandomAccessibleInterval<DoubleType> dx = input.derivedGauss(sigma, order(3, 0));
		RandomAccessibleInterval<DoubleType> dy = input.derivedGauss(sigma, order(3, 1));
		RandomAccessibleInterval<DoubleType> dz = input.derivedGauss(sigma, order(3, 2));
		LoopBuilder.setImages(dx, dy, dz, output.get(0)).multiThreaded().forEachPixel(
			(x, y, z, sum) -> sum.setReal(x.getRealDouble() + y.getRealDouble() + z.getRealDouble()));
	}

	private int[] order(int n, int d) {
		return IntStream.range(0, n).map(i -> i == d ? 2 : 0).toArray();
	}

	@Override
	public void prefetch(GpuFeatureInput input) {
		for (int d = 0; d < globalSettings().numDimensions(); d++)
			input.prefetchSecondDerivative(sigma, d, d, input.targetInterval());
	}

	@Override
	public void apply(GpuFeatureInput input, List<GpuView> output) {
		boolean is3d = globalSettings().numDimensions() == 3;
		GpuApi gpu = input.gpuApi();
		GpuPixelWiseOperation loopBuilder = GpuPixelWiseOperation.gpu(gpu);
		loopBuilder.addInput("dxx", input.secondDerivative(sigma, 0, 0, input.targetInterval()));
		loopBuilder.addInput("dyy", input.secondDerivative(sigma, 1, 1, input.targetInterval()));
		if (is3d)
			loopBuilder.addInput("dzz", input.secondDerivative(sigma, 2, 2, input.targetInterval()));
		loopBuilder.addOutput("output", output.get(0));
		String operation = is3d ? "output = dxx + dyy + dzz" : "output = dxx + dyy";
		loopBuilder.forEachPixel(operation);
	}
}
