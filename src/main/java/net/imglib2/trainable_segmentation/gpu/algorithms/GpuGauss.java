
package net.imglib2.trainable_segmentation.gpu.algorithms;

import net.imglib2.trainable_segmentation.gpu.api.GpuApi;
import net.imglib2.algorithm.convolution.kernel.Kernel1D;
import net.imglib2.algorithm.gauss3.Gauss3;

import java.util.ArrayList;
import java.util.List;

public class GpuGauss {

	public static GpuNeighborhoodOperation gauss(GpuApi gpu, double... sigmas) {
		final List<GpuKernelConvolution> convolutions = new ArrayList<>();
		for (int d = 0; d < sigmas.length; d++)
			convolutions.add(new GpuKernelConvolution(gpu, gaussKernel(sigmas[d]), d));
		return GpuNeighborhoodOperations.concat(gpu, convolutions);
	}

	private static Kernel1D gaussKernel(double sigma) {
		return Kernel1D.symmetric(Gauss3.halfkernels(new double[] { sigma })[0]);
	}
}
