
package net.imglib2.trainable_segmentation.pixel_feature.filter.stats;

import ij.ImagePlus;
import ij.ImageStack;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.img.planar.PlanarImgFactory;
import net.imglib2.test.RandomImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.StopWatch;
import net.imglib2.view.Views;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import net.imglib2.algorithm.convolution.kernel.Kernel1D;
import net.imglib2.algorithm.convolution.kernel.SeparableKernelConvolution;

import java.util.Arrays;

/**
 * Benchmarks {@link SumFilter} against {@link SeparableKernelConvolution}.
 */

@Fork(1)
@State(Scope.Benchmark)
@Warmup(iterations = 4)
@Measurement(iterations = 4)
@BenchmarkMode(Mode.AverageTime)
public class SumFilterBenchmark {

	private Img<FloatType> input = RandomImgs.seed(42).nextImage(new FloatType(), 100, 100, 100);
	private Img<FloatType> output = ArrayImgs.floats(100, 100, 100);

	private final int windowSize = 7;

	private ImageStack randomize(ImageStack stack) {
		RandomImgs.seed(42).randomize(ImageJFunctions.wrapFloat(new ImagePlus("", stack)));
		return stack;
	}

	@Benchmark
	public void benchmarkSumFilter() {
		runSumFilter(input, output);
	}

	@Benchmark
	public void benchmarkSeparableConvolution() {
		double[] ones = new double[windowSize];
		Arrays.fill(ones, 1.0);
		Kernel1D kernel = Kernel1D.centralAsymmetric(ones);
		SeparableKernelConvolution.convolution(kernel, kernel, kernel).process(Views.extendBorder(
			input), output);
	}

	private <T extends RealType<?>> void runSumFilter(RandomAccessibleInterval<T> input,
		RandomAccessibleInterval<T> output)
	{
		MinMaxFilter.maxFilter(windowSize, windowSize, windowSize).process(Views.extendBorder(input),
			output);
	}

	@Setup
	public void warmup() {
		for (int i = 0; i < 10; i++) {
			runSumFilter(new ArrayImgFactory<>(new LongType()));
			runSumFilter(new ArrayImgFactory<>(new IntType()));
			runSumFilter(new ArrayImgFactory<>(new DoubleType()));
			runSumFilter(new ArrayImgFactory<>(new FloatType()));
			runSumFilter(new ArrayImgFactory<>(new UnsignedByteType()));
			runSumFilter(new PlanarImgFactory<>(new UnsignedByteType()));
			runSumFilter(new CellImgFactory<>(new UnsignedByteType()));
		}
	}

	private <T extends RealType<T> & NativeType<T>> void runSumFilter(ImgFactory<T> factory) {
		RandomAccessibleInterval<T> input = RandomImgs.seed(42).randomize(factory.create(100, 100,
			100));
		RandomAccessibleInterval<T> output = factory.create(100, 100, 100);
		StopWatch stopWatch = StopWatch.createAndStart();
		runSumFilter(input, output);
		System.out.println(factory.getClass().getSimpleName() + "," + factory.type().getClass()
			.getSimpleName() + ": " + stopWatch);
	}

	public static void main(String... args) throws RunnerException {
		Options options = new OptionsBuilder().include(SumFilterBenchmark.class.getSimpleName())
			.build();
		new Runner(options).run();
	}

}
