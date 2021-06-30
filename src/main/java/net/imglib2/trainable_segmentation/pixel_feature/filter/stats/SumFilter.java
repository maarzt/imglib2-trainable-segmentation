
package net.imglib2.trainable_segmentation.pixel_feature.filter.stats;

import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import net.imglib2.algorithm.convolution.Convolution;
import net.imglib2.algorithm.convolution.LineConvolution;

import java.util.ArrayList;
import java.util.List;

class SumFilter {

	public static <T extends RealType<?>> Convolution<T> convolution(int... sizes) {
		List<Convolution<T>> convolutions = new ArrayList<>(sizes.length);
		for (int d = 0; d < sizes.length; d++)
			convolutions.add(new LineConvolution<>(SumConvolver.factory(sizes[d] / 2, (sizes[d] - 1) / 2),
				d));
		return Convolution.concat(convolutions);
	}

	public static void convolve(int[] sizes, RandomAccessible<? extends RealType<?>> source,
		RandomAccessibleInterval<? extends RealType<?>> target)
	{
		convolution(sizes).process(source, target);
	}
}
