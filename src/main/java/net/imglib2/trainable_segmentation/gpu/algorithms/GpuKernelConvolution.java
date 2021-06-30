
package net.imglib2.trainable_segmentation.gpu.algorithms;

import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.converter.RealTypeConverters;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.trainable_segmentation.gpu.api.GpuApi;
import net.imglib2.trainable_segmentation.gpu.api.GpuImage;
import net.imglib2.trainable_segmentation.gpu.api.GpuScope;
import net.imglib2.trainable_segmentation.gpu.api.GpuView;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.algorithm.convolution.kernel.Kernel1D;

import java.util.HashMap;

public class GpuKernelConvolution implements GpuNeighborhoodOperation {

	private final GpuApi gpu;

	private final Kernel1D kernel;

	private final int d;

	public GpuKernelConvolution(GpuApi gpu, Kernel1D kernel, int d) {
		this.gpu = gpu;
		this.kernel = kernel;
		this.d = d;
	}

	@Override
	public Interval getRequiredInputInterval(Interval targetInterval) {
		final long[] min = Intervals.minAsLongArray(targetInterval);
		final long[] max = Intervals.maxAsLongArray(targetInterval);
		min[d] += kernel.min();
		max[d] += kernel.max();
		return new FinalInterval(min, max);
	}

	@Override
	public void apply(GpuView input, GpuView output) {
		try (GpuApi scope = gpu.subScope()) {
			final Img<FloatType> kernelImage = ArrayImgs.floats(floats(kernel.fullKernel()), kernel
				.size());
			GpuImage kernelBuffer = scope.push(kernelImage);
			convolve(scope, kernelBuffer, input, output, d);
		}
	}

	private float[] floats(double[] doubles) {
		float[] floats = new float[doubles.length];
		for (int i = 0; i < doubles.length; i++)
			floats[i] = (float) doubles[i];
		return floats;
	}

	static void convolve(GpuApi gpu, GpuImage kernel, GpuView input, GpuView output, int d) {
		HashMap<String, Object> parameters = new HashMap<>();
		parameters.put("kernelValues", kernel);
		GpuSeparableOperation.run(gpu, "convolve1d.cl", kernel.getWidth(), parameters, input, output,
			d);
	}
}
