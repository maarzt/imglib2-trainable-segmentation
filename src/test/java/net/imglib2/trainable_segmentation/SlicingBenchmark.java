
package net.imglib2.trainable_segmentation;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.test.RandomImgs;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import net.imglib2.algorithm.gauss3.Gauss3;

import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@State(Scope.Benchmark)
@Warmup(iterations = 5, time = 100, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 10, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(1)
public class SlicingBenchmark {

	List<RandomAccessibleInterval<FloatType>> slices = IntStream.range(0, 10).mapToObj(i -> RandomImgs
		.seed(i).nextImage(new FloatType(), 100, 100, 100)).collect(Collectors.toList());

	RandomAccessibleInterval<FloatType> output = ArrayImgs.floats(100, 100, 100);

	@Benchmark
	public Object benchmarkWithSlicing() {
		RandomAccessibleInterval<FloatType> slice = Views.hyperSlice(Views.stack(slices), 3, 5);
		return blur(slice);
	}

	@Benchmark
	public Object benchmarkWithoutSlicing() {
		return blur(slices.get(5));
	}

	private RandomAccessibleInterval<FloatType> blur(RandomAccessibleInterval<FloatType> slice) {
		Gauss3.gauss(5.0, Views.extendBorder(slice), output);
		return output;
	}

	public static void main(String... args) throws RunnerException {
		Options options = new OptionsBuilder().include(SlicingBenchmark.class.getSimpleName())
			.build();
		new Runner(options).run();
	}
}
