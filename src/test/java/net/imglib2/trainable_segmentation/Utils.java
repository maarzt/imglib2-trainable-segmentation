
package net.imglib2.trainable_segmentation;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import net.imagej.ImageJ;
import net.imagej.ops.OpEnvironment;
import net.imagej.ops.OpService;
import net.imglib2.*;
import net.imglib2.cache.img.CachedCellImg;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.img.ImagePlusAdapter;
import net.imglib2.img.Img;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.test.ImgLib2Assert;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.trainable_segmentation.utils.DoubleTernaryOperator;
import net.imglib2.type.NativeType;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.ConstantUtils;
import net.imglib2.util.Intervals;
import net.imglib2.util.Localizables;
import net.imglib2.util.Pair;
import net.imglib2.util.Util;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.script.ScriptService;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.parallel.Parallelization;
import net.imglib2.parallel.TaskExecutor;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.StringJoiner;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.DoubleBinaryOperator;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.fail;

/**
 * @author Matthias Arzt
 */
public class Utils {

	public static void assertImagesEqual(ImagePlus expected, ImagePlus actual) {
		assertTrue(diffImagePlus(expected, actual) == 0);
	}

	private static int diffImagePlus(final ImagePlus a, final ImagePlus b) {
		final int[] dimsA = a.getDimensions(), dimsB = b.getDimensions();
		if (dimsA.length != dimsB.length) return dimsA.length - dimsB.length;
		for (int i = 0; i < dimsA.length; i++) {
			if (dimsA[i] != dimsB[i]) return dimsA[i] - dimsB[i];
		}
		int count = 0;
		final ImageStack stackA = a.getStack(), stackB = b.getStack();
		for (int slice = 1; slice <= stackA.getSize(); slice++) {
			count += diff(stackA.getProcessor(slice), stackB.getProcessor(slice));
		}
		return count;
	}

	private static int diff(final ImageProcessor a, final ImageProcessor b) {
		int count = 0;
		final int width = a.getWidth(), height = a.getHeight();
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (a.getf(x, y) != b.getf(x, y)) count++;
			}
		}
		return count;
	}

	public static ImagePlus loadImagePlusFromResource(final String path) {
		ImagePlus image = new ImagePlus(path);
		if (image.getWidth() != 0)
			return image;
		final URL url = Utils.class.getResource("/" + path);
		if (url == null)
			throw new NoSuchElementException("file: " + path);
		if ("file".equals(url.getProtocol())) return new ImagePlus(url.getPath());
		return new ImagePlus(url.toString());
	}

	public static void saveImageToResouce(final ImagePlus image, final String path) {
		final URL url = Utils.class.getResource(path);
		IJ.save(image, url.getPath());
	}

	public static <A extends Type<A>>
		void assertImagesEqual(final RandomAccessibleInterval<? extends A> a,
			final RandomAccessibleInterval<? extends A> b)
	{
		ImgLib2Assert.assertImageEquals(a, b);
	}

	public static void assertImagesEqual(final ImagePlus expected,
		final RandomAccessibleInterval<FloatType> actual)
	{
		assertImagesEqual(ImagePlusAdapter.convertFloat(expected), actual);
	}

	public static void assertImagesEqual(final ImageProcessor expected,
		final RandomAccessibleInterval<FloatType> actual)
	{
		assertImagesEqual(new ImagePlus("expected", expected), actual);
	}

	private static <A extends Type<A>> String positionString(Localizable cursor) {
		StringJoiner joiner = new StringJoiner(", ");
		for (int i = 0, n = cursor.numDimensions(); i < n; i++)
			joiner.add(String.valueOf(cursor.getIntPosition(i)));
		return "(" + joiner + ")";
	}

	public static <T extends NumericType<T>> void showDifference(
		RandomAccessibleInterval<T> expectedImage, RandomAccessibleInterval<T> resultImage)
	{
		ImgLib2Assert.assertIntervalEquals(expectedImage, resultImage);
		ImagePlus window = ImageJFunctions.show(tile(expectedImage, resultImage, subtract(expectedImage,
			resultImage)));
		try {
			while (window.isVisible())
				Thread.sleep(100);
		}
		catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	public static void show(Object... images) {
		ImageJ imageJ = new ImageJ();
		imageJ.ui().showUI();
		for (Object o : images)
			imageJ.ui().show(translateToOrigin(o));
	}

	private static Object translateToOrigin(Object o) {
		if (o instanceof RandomAccessibleInterval)
			return Views.zeroMin((RandomAccessibleInterval<?>) o);
		return o;
	}

	public static RandomAccessibleInterval<IntType> loadImageIntType(String s) {
		RandomAccessibleInterval<UnsignedByteType> img = ImagePlusAdapter.wrapByte(loadImage(s));
		return Converters.convert(img, (b, i) -> i.set(b.get()), new IntType());
	}

	public static RandomAccessibleInterval<FloatType> loadImageFloatType(String s) {
		return Objects.requireNonNull(ImagePlusAdapter.convertFloat(loadImage(s)));
	}

	public static ImagePlus loadImage(String s) {
		if (new File(s).exists())
			return new ImagePlus(s);
		return Utils.loadImagePlusFromResource(s);
	}

	public static double psnr(RandomAccessibleInterval<? extends ComplexType<?>> expected,
		RandomAccessibleInterval<? extends ComplexType<?>> actual)
	{
		double meanSquareError = meanSquareError(expected, actual);
		if (meanSquareError == 0.0)
			return Float.POSITIVE_INFINITY;
		return (20 * Math.log10(maxAbs(expected)) - 10 * Math.log10(meanSquareError));
	}

	private static double meanSquareError(RandomAccessibleInterval<? extends ComplexType<?>> a,
		RandomAccessibleInterval<? extends ComplexType<?>> b)
	{
		if (!Intervals.equals(a, b))
			throw new IllegalArgumentException("both arguments must be the same interval" +
				"given: " + showInterval(a) + " and: " + showInterval(b));
		DoubleType sum = new DoubleType(0.0f);
		Views.interval(Views.pair(a, b), a).forEach(x -> sum.set(sum.get() + sqr(x.getA()
			.getRealDouble() - x.getB().getRealDouble())));
		return sum.get() / Intervals.numElements(a);
	}

	private static String showInterval(Interval b) {
		StringJoiner j = new StringJoiner(", ");
		int n = b.numDimensions();
		for (int i = 0; i < n; i++)
			j.add(b.min(i) + " - " + b.max(i));
		return "[" + j + "]";
	}

	private static double sqr(double v) {
		return v * v;
	}

	private static double maxAbs(RandomAccessibleInterval<? extends ComplexType<?>> a) {
		double result = 0;
		for (ComplexType<?> pixel : Views.iterable(a))
			result = Math.max(result, Math.abs(pixel.getRealDouble()));
		return result;
	}

	public static void showPsnr(RandomAccessibleInterval<FloatType> expected,
		RandomAccessibleInterval<FloatType> actual)
	{
		System.out.println("psnr: " + psnr(expected, actual));
	}

	public static ImagePlus createImage(final String title, final int width, final int height,
		final int... pixels)
	{
		assertEquals(pixels.length, width * height);
		final byte[] bytes = new byte[pixels.length];
		for (int i = 0; i < bytes.length; i++)
			bytes[i] = (byte) pixels[i];
		final ByteProcessor bp = new ByteProcessor(width, height, bytes, null);
		return new ImagePlus(title, bp);
	}

	public static ImagePlus createImage(String title, int width, int height, final float... pixels) {
		assertEquals(pixels.length, width * height);
		final FloatProcessor processor = new FloatProcessor(width, height, pixels.clone());
		return new ImagePlus(title, processor);
	}

	public static <T> String pixelsAsString(RandomAccessibleInterval<T> image) {
		StringJoiner joiner = new StringJoiner(", ");
		for (T pixel : Views.iterable(image))
			joiner.add(pixel.toString());
		return "[" + joiner.toString() + "]";
	}

	public static void assertImagesEqual(double expectedPsnr,
		RandomAccessibleInterval<? extends ComplexType<?>> expected,
		RandomAccessibleInterval<? extends ComplexType<?>> actual)
	{
		double psnr = Utils.psnr(expected, actual);
		if (RevampUtils.containsNaN(expected))
			fail("Cannot calculate PSNR because expected picture contains NaN value.");
		if (RevampUtils.containsNaN(actual))
			fail("Cannot calculate PSNR because actual picture contains NaN value.");
		if (psnr < expectedPsnr)
			fail("Actual PSNR is lower than expected. Actual: " + psnr + " Expected: " + expectedPsnr);
	}

	public static <T extends NumericType<T>> RandomAccessibleInterval<T> subtract(
		RandomAccessibleInterval<T> expected, RandomAccessibleInterval<T> result)
	{
		RandomAccessibleInterval<Pair<T, T>> interval = Views.interval(Views.pair(expected, result),
			result);
		return Converters.convert(interval, (p, out) -> {
			out.set(p.getA());
			out.sub(p.getB());
		}, Util.getTypeFromInterval(expected).createVariable());
	}

	public static RandomAccessibleInterval<IntType> toInt(RandomAccessibleInterval<FloatType> input) {
		Converter<FloatType, IntType> floatToInt = (in, out) -> out.set((int) in.get());
		return Converters.convert(input, floatToInt, new IntType());
	}

	private static <T extends Type<T>> RandomAccessibleInterval<T> tile(
		RandomAccessibleInterval<T>... imgs)
	{
		long[] size = Intervals.dimensionsAsLongArray(imgs[0]);
		final T type = Util.getTypeFromInterval(imgs[0]);
		long[] outputSize = size.clone();
		outputSize[0] *= imgs.length;
		Img<T> out = (Img<T>) new ArrayImgFactory<>((NativeType) type).create(outputSize);
		for (int i = 0; i < imgs.length; i++)
			copy(imgs[i], Views.interval(out, Intervals.translate(new FinalInterval(size), size[0] * i,
				0)));
		return out;
	}

	private static <T extends Type<T>> void copy(RandomAccessibleInterval<T> src,
		RandomAccessibleInterval<T> target)
	{
		LoopBuilder.setImages(src, target).multiThreaded().forEachPixel((i, o) -> o.set(i));
	}

	public static double gauss(double sigma, double x) {
		double factor = 1 / Math.sqrt(2 * Math.PI * square(sigma));
		return factor * Math.exp(-0.5 / square(sigma) * square(x));
	}

	public static double gauss(double sigma, double x, double y) {
		return gauss(sigma, x) * gauss(sigma, y);
	}

	public static double gauss(double sigma, double x, double y, double z) {
		return gauss(sigma, x) * gauss(sigma, y) * gauss(sigma, z);
	}

	public static double square(double x) {
		return x * x;
	}

	public static RandomAccessibleInterval<FloatType> create2dImage(Interval interval,
		DoubleBinaryOperator function)
	{
		RandomAccessibleInterval<Localizable> positions = Views.interval(Localizables.randomAccessible(
			2), interval);
		Converter<Localizable, FloatType> converter = (i, o) -> o.setReal(function.applyAsDouble(i
			.getDoublePosition(0), i.getDoublePosition(1)));
		return Converters.convert(positions, converter, new FloatType());
	}

	public static RandomAccessibleInterval<FloatType> create3dImage(Interval interval,
		DoubleTernaryOperator function)
	{
		RandomAccessibleInterval<Localizable> positions = Views.interval(Localizables.randomAccessible(
			3), interval);
		Converter<Localizable, FloatType> converter = (i, o) -> o.setReal(function.applyAsDouble(i
			.getDoublePosition(0), i.getDoublePosition(1), i.getDoublePosition(2)));
		return Converters.convert(positions, converter, new FloatType());
	}

	public static RandomAccessible<FloatType> dirac2d() {
		return dirac(2);
	}

	public static RandomAccessible<FloatType> dirac(int n) {
		RandomAccessibleInterval<FloatType> floats = ConstantUtils.constantRandomAccessibleInterval(
			new FloatType(1), new FinalInterval(new long[n], new long[n]));
		return Views.extendZero(floats);
	}

	public static Img<ARGBType> loadImageARGBType(String pathOrURL) {
		return ImageJFunctions.wrapRGBA(new ImagePlus(
			pathOrURL));
	}

	public static void populateCellImg(CachedCellImg<UnsignedShortType, ?> segmentation) {
		TaskExecutor executor = Parallelization.getTaskExecutor();
		List<Point> cellCenters = getCellCenters(segmentation.getCellGrid());
		AtomicInteger counter = new AtomicInteger();
		executor.forEach(cellCenters, cellCenter -> {
			RandomAccess<UnsignedShortType> ra = segmentation.randomAccess();
			ra.setPosition(cellCenter);
			ra.get();
			System.out.println("step " + counter.incrementAndGet() + "/" + cellCenters.size());
		});
	}

	private static List<Point> getCellCenters(CellGrid grid) {
		List<Point> cells = new ArrayList<>();
		long numCells = Intervals.numElements(grid.getGridDimensions());
		for (int i = 0; i < numCells; i++) {
			long[] cellMin = new long[3];
			int[] cellDims = new int[3];
			grid.getCellDimensions(i, cellMin, cellDims);
			for (int d = 0; d < cellMin.length; d++)
				cellMin[d] += cellDims[d] / 2;
			cells.add(new Point(cellMin));
		}
		return cells;
	}
}
