
package net.imglib2.trainable_segmentation;

import net.imglib2.*;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.converter.RealTypeConverters;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.trainable_segmentation.utils.views.FastViews;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;
import net.imglib2.util.Intervals;
import net.imglib2.view.StackView;
import net.imglib2.view.Views;
import net.imglib2.view.composite.Composite;
import net.imglib2.algorithm.convolution.kernel.Kernel1D;
import net.imglib2.algorithm.convolution.kernel.SeparableKernelConvolution;
import weka.core.DenseInstance;
import net.imglib2.algorithm.gauss3.Gauss3;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

/**
 * @author Matthias Arzt
 */
public class RevampUtils {

	public static <T> List<RandomAccessibleInterval<T>> slices(RandomAccessibleInterval<T> output) {
		int axis = output.numDimensions() - 1;
		return LongStream.range(output.min(axis), output.max(axis) + 1)
			.mapToObj(pos -> Views.hyperSlice(output, axis, pos))
			.collect(Collectors.toList());
	}

	public static <T extends NativeType<T>> RandomAccessibleInterval<T> createImage(Interval interval,
		T type)
	{
		long[] min = Intervals.minAsLongArray(interval);
		long[] size = Intervals.dimensionsAsLongArray(interval);
		return Views.translate(new ArrayImgFactory<>(type).create(size), min);
	}

	public static long[] extend(long[] in, long elem) {
		long result[] = new long[in.length + 1];
		System.arraycopy(in, 0, result, 0, in.length);
		result[in.length] = elem;
		return result;
	}

	public static int[] extend(int[] in, int elem) {
		int result[] = new int[in.length + 1];
		System.arraycopy(in, 0, result, 0, in.length);
		result[in.length] = elem;
		return result;
	}

	// TODO: move to Intervals?
	public static Interval appendDimensionToInterval(Interval in, long min, long max) {
		int n = in.numDimensions();
		long[] mins = new long[n + 1];
		long[] maxs = new long[n + 1];
		for (int i = 0; i < n; i++) {
			mins[i] = in.min(i);
			maxs[i] = in.max(i);
		}
		mins[n] = min;
		maxs[n] = max;
		return new FinalInterval(mins, maxs);
	}

	public static Interval removeLastDimension(Interval in) {
		long[] min = removeLast(Intervals.minAsLongArray(in));
		long[] max = removeLast(Intervals.maxAsLongArray(in));
		return new FinalInterval(min, max);
	}

	private static long[] removeLast(long[] longs) {
		return Arrays.copyOf(longs, longs.length - 1);
	}

	public static RandomAccessibleInterval<FloatType> gauss(RandomAccessibleInterval<FloatType> image,
		double[] sigmas)
	{
		return gauss(Views.extendBorder(image), (Interval) image, sigmas);
	}

	public static RandomAccessibleInterval<FloatType> gauss(RandomAccessible<FloatType> input,
		Interval outputInterval, double[] sigmas)
	{
		RandomAccessibleInterval<FloatType> blurred = RevampUtils.createImage(outputInterval,
			new FloatType());
		Gauss3.gauss(sigmas, input, blurred);
		return blurred;
	}

	public static Interval gaussRequiredInput(Interval outputInterval, double[] sigmas) {
		long[] border = IntStream.of(Gauss3.halfkernelsizes(sigmas))
			.mapToLong(x -> x - 1)
			.toArray();
		return Intervals.expand(outputInterval, border);
	}

	public static RandomAccessibleInterval<FloatType> deriveX(RandomAccessible<FloatType> input,
		Interval outputInterval)
	{
		if (outputInterval.numDimensions() != 2)
			throw new IllegalArgumentException("Only two dimensional images supported.");
		RandomAccessibleInterval<FloatType> output = RevampUtils.createImage(outputInterval,
			new FloatType());
		SeparableKernelConvolution.convolution(Kernel1D.centralAsymmetric(1, 0, -1), Kernel1D
			.centralAsymmetric(1, 2, 1))
			.process(input, output);
		return output;
	}

	public static Interval deriveXRequiredInput(Interval output) {
		if (output.numDimensions() != 2)
			throw new IllegalArgumentException("Only two dimensional images supported.");
		return Intervals.expand(output, new long[] { 1, 1 });
	}

	public static RandomAccessibleInterval<FloatType> deriveY(RandomAccessible<FloatType> input,
		Interval outputInterval)
	{
		if (outputInterval.numDimensions() != 2)
			throw new IllegalArgumentException("Only two dimensional images supported.");
		RandomAccessibleInterval<FloatType> output = RevampUtils.createImage(outputInterval,
			new FloatType());
		SeparableKernelConvolution.convolution(Kernel1D.centralAsymmetric(1, 2, 1), Kernel1D
			.centralAsymmetric(1, 0, -1))
			.process(input, output);
		return output;
	}

	public static Interval deriveYRequiredInput(Interval output) {
		if (output.numDimensions() != 2)
			throw new IllegalArgumentException("Only two dimensional images supported.");
		return Intervals.expand(output, new long[] { 1, 1 });
	}

	public static RandomAccessible<FloatType> randomAccessibleToFloat(
		RandomAccessible<? extends RealType<?>> input)
	{
		RealType<?> inputType = input.randomAccess().get();
		if (inputType instanceof FloatType)
			return uncheckedCast(input);
		Converter<RealType<?>, FloatType> converter = RealTypeConverters.getConverter(inputType,
			new FloatType());
		return Converters.convert(input, converter, new FloatType());
	}

	public static RandomAccessibleInterval<FloatType> copy(
		RandomAccessibleInterval<FloatType> input)
	{
		RandomAccessibleInterval<FloatType> output = RevampUtils.createImage(input, new FloatType());
		RealTypeConverters.copyFromTo(input, output);
		return output;
	}

	public static boolean containsNaN(RandomAccessibleInterval<? extends ComplexType<?>> result) {
		for (ComplexType<?> value : Views.iterable(result))
			if (Double.isNaN(value.getRealDouble()))
				return true;
		return false;
	}

	public static long[] nCopies(int count, long value) {
		long[] result = new long[count];
		Arrays.fill(result, value);
		return result;
	}

	public static void wrapException(RunnableWithException r) {
		try {
			r.run();
		}
		catch (Exception e) {
			if (e instanceof RuntimeException)
				throw (RuntimeException) e;
			throw new RuntimeException(e);
		}
	}

	public static <R> R wrapException(SupplierWithException<R> r) {
		try {
			return r.get();
		}
		catch (Exception e) {
			if (e instanceof RuntimeException)
				throw (RuntimeException) e;
			throw new RuntimeException(e);
		}
	}

	public static DenseInstance getInstance(int featureCount, int classIndex,
		Composite<? extends RealType<?>> featureValues)
	{
		double[] values = new double[featureCount + 1];
		for (int i = 0; i < featureCount; i++)
			values[i] = featureValues.get(i).getRealDouble();
		values[featureCount] = classIndex;
		return new DenseInstance(1.0, values);
	}

	public static <T> T uncheckedCast(Object input) {
		@SuppressWarnings("unchecked")
		T result = (T) input;
		return result;
	}

	public static <T> RandomAccessibleInterval<Composite<T>> vectorizeStack(
		List<RandomAccessibleInterval<T>> derivatives)
	{
		return Cast.unchecked(FastViews.collapse(Views.stack(
			StackView.StackAccessMode.MOVE_ALL_SLICE_ACCESSES, derivatives)));
	}

	public interface RunnableWithException {

		void run() throws Exception;
	}

	public interface SupplierWithException<R> {

		R get() throws Exception;
	}

	public static Interval intervalRemoveDimension(Interval interval) {
		return intervalRemoveDimension(interval, interval.numDimensions() - 1);
	}

	public static Interval intervalRemoveDimension(Interval interval,
		int d)
	{
		long[] min = removeElement(Intervals.minAsLongArray(interval), d);
		long[] max = removeElement(Intervals.maxAsLongArray(interval), d);
		return new FinalInterval(min, max);
	}

	private static long[] removeElement(long[] values, int d) {
		long[] result = new long[values.length - 1];
		System.arraycopy(values, 0, result, 0, d);
		System.arraycopy(values, d + 1, result, d, values.length - d - 1);
		return result;
	}
}
