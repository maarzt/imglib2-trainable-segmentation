
package net.imglib2.trainable_segmentation;

import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.trainable_segmentation.pixel_feature.calculator.FeatureCalculator;
import net.imglib2.trainable_segmentation.pixel_feature.filter.GroupedFeatures;
import net.imglib2.trainable_segmentation.pixel_feature.settings.FeatureSetting;
import net.imglib2.trainable_segmentation.pixel_feature.settings.FeatureSettings;
import net.imglib2.trainable_segmentation.pixel_feature.settings.GlobalSettings;
import net.imglib2.trainable_segmentation.utils.SingletonContext;
import net.imglib2.type.numeric.real.FloatType;
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
import org.scijava.Context;
import net.imglib2.parallel.Parallelization;

import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
@Warmup(iterations = 5, time = 100, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 100, timeUnit = TimeUnit.MILLISECONDS)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(1)
public class FeatureCalculationBenchmark {

	private static final String imageFilename =
		"drosophila_3d.tif";

	private static final String labelingFileName =
		"drosophila_3d_labeling.tif";

	private final Context context = SingletonContext.getInstance();

	private final RandomAccessibleInterval<FloatType> image = Utils.loadImageFloatType(imageFilename);

	@Benchmark
	public Object gauss() {
		return calculateFeature(GroupedFeatures.gauss());
	}

	@Benchmark
	public Object differenceOfGaussians() {
		return calculateFeature(GroupedFeatures.differenceOfGaussians());
	}

	@Benchmark
	public Object hessian3D() {
		return calculateFeature(GroupedFeatures.hessian());
	}

	@Benchmark
	public Object gradient() {
		return calculateFeature(GroupedFeatures.gradient());
	}

	@Benchmark
	public Object min() {
		return calculateFeature(GroupedFeatures.min());
	}

	@Benchmark
	public Object max() {
		return calculateFeature(GroupedFeatures.max());
	}

	@Benchmark
	public Object mean() {
		return calculateFeature(GroupedFeatures.mean());
	}

	@Benchmark
	public Object variance() {
		return calculateFeature(GroupedFeatures.variance());
	}

	@Benchmark
	public Object structureTensor() {
		return calculateFeature(GroupedFeatures.structureTensor());
	}

	private Object calculateFeature(FeatureSetting setting) {
		return Parallelization.runSingleThreaded(() -> {
			final FeatureSettings featureSettings = new FeatureSettings(GlobalSettings.default3d()
				.build(), setting);
			FeatureCalculator featureCalculator = new FeatureCalculator(context, featureSettings);
			return featureCalculator.apply(image);
		});
	}

	public static void main(String... args) throws RunnerException {
		runBenchmark();
	}

	private static void runBenchmark() throws RunnerException {
		Options options = new OptionsBuilder().include(FeatureCalculationBenchmark.class
			.getSimpleName()).build();
		new Runner(options).run();
	}

}
