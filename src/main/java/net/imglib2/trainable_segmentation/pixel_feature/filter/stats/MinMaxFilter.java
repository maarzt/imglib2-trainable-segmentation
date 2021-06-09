
package net.imglib2.trainable_segmentation.pixel_feature.filter.stats;

import net.imglib2.RandomAccess;
import net.imglib2.type.numeric.RealType;
import net.imglib2.algorithm.convolution.Convolution;
import net.imglib2.algorithm.convolution.LineConvolution;
import net.imglib2.algorithm.convolution.LineConvolverFactory;
import net.imglib2.loops.ClassCopyProvider;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringJoiner;

class MinMaxFilter {

	public enum Operation {
			MIN, MAX
	}

	/**
	 * Returns a {@link Convolution}, that performs a max pooling.
	 *
	 * @param windowSize specifies the window size.
	 */
	public static <T extends RealType<?>> Convolution<T> maxFilter(int... windowSize) {
		return minMaxFilter(Operation.MAX, windowSize);
	}

	/**
	 * Returns a {@link Convolution}, that performs a min pooling.
	 *
	 * @param windowSize specifies the window size.
	 */
	public static <T extends RealType<?>> Convolution<T> minFilter(int... windowSize) {
		return minMaxFilter(Operation.MIN, windowSize);
	}

	private static <T extends RealType<?>> Convolution<T> minMaxFilter(Operation operation,
		int[] windowSize)
	{
		List<Convolution<T>> convolutions = new ArrayList<>();
		for (int d = 0; d < windowSize.length; d++) {
			int radius = windowSize[d];
			if (radius != 0)
				convolutions.add(minMaxFilter1d(operation, radius, d));
		}
		return Convolution.concat(convolutions);
	}

	/**
	 * Returns a {@link Convolution} that performs a max pooling in one dimension.
	 * Specified window size in the given dimension d, and 1 pixel in every other
	 * direction.
	 */
	public static <T extends RealType<?>> Convolution<T> maxFilter1d(int windowSize, int d) {
		return minMaxFilter1d(Operation.MAX, windowSize, d);
	}

	/**
	 * Returns a {@link Convolution} that performs a min pooling in one dimension.
	 * Specified window size in the given dimension d, and 1 pixel in every other
	 * direction.
	 */
	public static <T extends RealType<?>> Convolution<T> minFilter1d(int windowSize, int d) {
		return minMaxFilter1d(Operation.MIN, windowSize, d);
	}

	private static <T extends RealType<?>> Convolution<T> minMaxFilter1d(Operation operation,
		int windowSize, int d)
	{
		return new LineConvolution<>(new CenteredMinMaxFilter<>(operation, windowSize), d);
	}

	/**
	 * This class, wraps around {@link MinMaxConvolver}. It can be used with
	 * {@link LineConvolution} to implement a (one dimensional) max pooling
	 * operation. Specified window size in the given dimension d, and 1 pixel in
	 * every other direction.
	 */
	static class CenteredMinMaxFilter<T extends RealType<?>> implements LineConvolverFactory<T> {

		private static final ClassCopyProvider<Runnable> provider =
			new ClassCopyProvider<>(MinMaxConvolver.class, Runnable.class);

		private final Operation operation;

		private final int windowSize;

		CenteredMinMaxFilter(Operation operation, int windowSize) {
			assert windowSize > 0;
			this.operation = operation;
			this.windowSize = windowSize;
		}

		@Override
		public long getBorderBefore() {
			return windowSize / 2;
		}

		@Override
		public long getBorderAfter() {
			return (windowSize - 1) / 2;
		}

		@Override
		public Runnable getConvolver(RandomAccess<? extends T> in, RandomAccess<? extends T> out, int d,
			long lineLength)
		{
			List<Object> key = Arrays.asList(operation, in.getClass(), out.getClass(), in.get()
				.getClass(), out.get().getClass());
			return provider.newInstanceForKey(key, operation, windowSize, in, out, d, lineLength);
		}

		@Override
		public T preferredSourceType(T targetType) {
			return targetType;
		}
	}

	/**
	 * This class is similar to
	 * {@link net.imglib2.algorithm.convolution.kernel.DoubleConvolverRealType} but
	 * it performs a max pooling operation instead of a convolution.
	 * <p>
	 * The algorithm's runtime is independent of the window size. The memory
	 * requirement is linear with the window size.
	 */
	public static class MinMaxConvolver implements Runnable {

		private final RandomAccess<? extends RealType<?>> in;

		private final RandomAccess<? extends RealType<?>> out;

		private final int width;

		private final long lineLength;

		private final LongDoubleQueue queue;

		private final int d;

		private final Operation operation;

		public MinMaxConvolver(Operation operation, int width, RandomAccess<? extends RealType<?>> in,
			RandomAccess<? extends RealType<?>> out, int d, long lineLength)
		{
			this.in = in;
			this.out = out;
			this.width = width;
			this.d = d;
			this.lineLength = lineLength;
			this.queue = new LongDoubleQueue(width);
			this.operation = operation;
		}

		@Override
		public void run() {
			queue.reset();
			prefill();
			process();
		}

		private void prefill() {
			for (long i = -width + 1; i < 0; i++) {
				step(i);
			}
		}

		private void process() {
			for (long i = 0; i < lineLength; i++) {
				step(i);
				write(queue.getFirstDouble());
			}
		}

		private void step(long i) {
			double value = read();
			if (!queue.isEmpty() && queue.getFirstLong() <= i)
				queue.removeFirst();
			while (!queue.isEmpty() && useFirst(value, queue.getLastDouble()))
				queue.removeLast();
			queue.addLast(i + width, value);
		}

		private boolean useFirst(double a, double b) {
			return operation == Operation.MAX ? a >= b : a <= b;
		}

		private double read() {
			double v = in.get().getRealDouble();
			in.fwd(d);
			return v;
		}

		private void write(double v) {
			out.get().setReal(v);
			out.fwd(d);
		}
	}

	/**
	 * {@link LongDoubleQueue} is a special double ended queue, used by
	 * {@link MinMaxConvolver}.
	 * <p>
	 * The elements of the queue are pairs. Each pair consists of a long value and a
	 * double value. The values are stored "unboxed".
	 * <p>
	 * The queue uses a circular buffer with a fixed size. The capacity of the queue
	 * is therefor fixed. Adding more elements to the queue will result in undefined
	 * behavior.
	 * <p>
	 * The queue supports the following operations with O(1) complexity:
	 * <ul>
	 * <li>Adding an element to the end of the que.</li>
	 * <li>Removing the first or last element.</li>
	 * <li>Accessing the first and last element.</li>
	 * </ul>
	 */
	public static class LongDoubleQueue {

		private final int capacity;

		private final long[] longs;

		private final double[] doubles;

		private int first;

		private int last;

		private int size;

		/**
		 * Create a {@link LongDoubleQueue}.
		 *
		 * @param capacity Highest number of elements, that can be hold by the queue.
		 */
		public LongDoubleQueue(int capacity) {
			this.capacity = capacity;
			this.longs = new long[capacity];
			this.doubles = new double[capacity];
			this.first = 0;
			this.last = capacity - 1;
			this.size = 0;
		}

		public void reset() {
			this.first = 0;
			this.last = capacity - 1;
			this.size = 0;
		}

		/**
		 * Returns the long value of the first element in the queue
		 */
		public long getFirstLong() {
			return longs[first];
		}

		/**
		 * Returns the double value of the first element in the queue
		 */
		public double getFirstDouble() {
			return doubles[first];
		}

		/**
		 * Returns the double value of the last element in the queue
		 */
		public double getLastDouble() {
			return doubles[last];
		}

		/**
		 * Adds an element to the end of the queue.
		 */
		public void addLast(long longValue, double doubleValue) {
			last = Math.floorMod(last + 1, capacity);
			longs[last] = longValue;
			doubles[last] = doubleValue;
			size++;
		}

		/**
		 * Removes the first element of the queue.
		 */
		public void removeFirst() {
			first = Math.floorMod(first + 1, capacity);
			size--;
		}

		/**
		 * Removes the last element of the queue.
		 */
		public void removeLast() {
			last = Math.floorMod(last - 1, capacity);
			size--;
		}

		/**
		 * Returns true is the queue is empty.
		 */
		public boolean isEmpty() {
			return size <= 0;
		}

		@Override
		public String toString() {
			StringJoiner joiner = new StringJoiner(", ", "[", "]");
			for (int i = 0; i < size; i++) {
				int index = Math.floorMod(first + i, capacity);
				joiner.add(longs[index] + ":" + doubles[index]);
			}
			return joiner.toString();
		}
	}
}
