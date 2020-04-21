
package net.imglib2.trainable_segmention.gpu.random_forest;

import net.imglib2.trainable_segmention.gpu.api.GpuApi;
import net.imglib2.trainable_segmention.gpu.api.GpuImage;
import net.imglib2.util.Intervals;

import java.util.HashMap;

public class GpuRandomForestKernel {

	public static void randomForest(GpuApi gpu,
		GpuImage distributions,
		GpuImage src,
		GpuImage thresholds,
		GpuImage probabilities,
		GpuImage indices,
		int numberOfFeatures)
	{
		long[] globalSizes = src.getDimensions().clone();
		HashMap<String, Object> parameters = new HashMap<>();
		parameters.put("src", src);
		parameters.put("dst", distributions);
		parameters.put("thresholds", thresholds);
		parameters.put("probabilities", probabilities);
		parameters.put("indices", indices);
		HashMap<String, Object> constants = new HashMap<>();
		constants.put("NUMBER_OF_CLASSES", probabilities.getWidth());
		constants.put("NUMBER_OF_FEATURES", numberOfFeatures);
		constants.put("INDICES_SIZE", Intervals.numElements(indices.getDimensions()));
		gpu.execute(GpuRandomForestKernel.class, "random_forest.cl", "random_forest", globalSizes, null,
			parameters, constants);
	}

	public static void findMax(GpuApi gpu,
		GpuImage distributions,
		GpuImage dst)
	{
		long[] globalSizes = { dst.getWidth(), dst.getHeight(), dst.getDepth() };
		HashMap<String, Object> parameters = new HashMap<>();
		parameters.put("dst", dst);
		parameters.put("src", distributions);
		parameters.put("num_classes", (int) distributions.getNumberOfChannels());
		gpu.execute(GpuRandomForestKernel.class, "find_max.cl", "find_max", globalSizes, null,
			parameters, null);
	}
}