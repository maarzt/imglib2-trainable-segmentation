
package net.imglib2.trainable_segmentation.pixel_feature.filter.stats;

import net.imglib2.RandomAccess;
import net.imglib2.type.numeric.RealType;
import net.imglib2.algorithm.convolution.LineConvolverFactory;

import java.util.Arrays;

class SumConvolver implements Runnable {

	static <T extends RealType<?>> LineConvolverFactory<T> factory(long before, long after) {
		return new LineConvolverFactory<T>() {

			@Override
			public long getBorderBefore() {
				return before;
			}

			@Override
			public long getBorderAfter() {
				return after;
			}

			@Override
			public Runnable getConvolver(RandomAccess<? extends T> in, RandomAccess<? extends T> out,
				int d, long lineLength)
			{
				return new SumConvolver(before + 1 + after, in, out, d, lineLength);
			}

			@Override
			public T preferredSourceType(T targetType) {
				return targetType;
			}
		};
	}

	private final RandomAccess<? extends RealType<?>> in;

	private final RandomAccess<? extends RealType<?>> out;

	private final int d;

	private final int width;

	private final double[] buffer;

	private final long length;

	public SumConvolver(long width, RandomAccess<? extends RealType<?>> in,
		RandomAccess<? extends RealType<?>> out,
		int d, long length)
	{
		this.in = in;
		this.out = out;
		this.d = d;
		if (width >= (long) Integer.MAX_VALUE - 8)
			throw new IllegalArgumentException(
				"Width of SumConvolver needs to be less than Integer.MAX - 8");
		this.width = (int) width;
		this.buffer = new double[this.width];
		this.length = length;
	}

	@Override
	public void run() {
		Arrays.fill(buffer, 0);
		double sumA = 0;
		for (int i = 1; i < width; i++) {
			double value = in.get().getRealDouble();
			in.fwd(d);
			buffer[i] = value;
			sumA += value;
		}
		double remaining = this.length;
		while (remaining > width) {
			double sumB = 0;
			for (int i = 0; i < width; i++) {
				sumA -= buffer[i];
				double value = in.get().getRealDouble();
				in.fwd(d);
				buffer[i] = value;
				sumB += value;
				out.get().setReal(sumA + sumB);
				out.fwd(d);
			}
			remaining -= width;
			sumA = sumB;
		}
		for (int i = 0; i < remaining; i++) {
			sumA += in.get().getRealDouble() - buffer[i];
			in.fwd(d);
			out.get().setReal(sumA);
			out.fwd(d);
		}
	}
}
