
package net.imglib2.trainable_segmentation;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.roi.labeling.ImgLabeling;
import net.imglib2.roi.labeling.LabelRegions;
import net.imglib2.roi.labeling.LabelingType;
import net.imglib2.trainable_segmentation.classification.Segmenter;
import net.imglib2.trainable_segmentation.classification.Trainer;
import net.imglib2.trainable_segmentation.pixel_feature.filter.GroupedFeatures;
import net.imglib2.trainable_segmentation.pixel_feature.settings.FeatureSettings;
import net.imglib2.trainable_segmentation.pixel_feature.settings.GlobalSettings;
import net.imglib2.trainable_segmentation.utils.SingletonContext;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.StopWatch;
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
import net.imglib2.loops.LoopBuilder;
import net.imglib2.parallel.Parallelization;

import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
@Warmup(iterations = 15, time = 100, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 10, time = 100, timeUnit = TimeUnit.MILLISECONDS)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(0)
public class SegmentationBenchmark {

	private static final String imageFilename =
		"drosophila_3d.tif";

	private static final String labelingFileName =
		"drosophila_3d_labeling.tif";

	private final Context context = SingletonContext.getInstance();

	private final RandomAccessibleInterval<FloatType> image = Utils.loadImageFloatType(imageFilename);

	private final LabelRegions<?> labelRegions = asLabelRegions(Utils.loadImageFloatType(
		labelingFileName));

	private final Segmenter segmenter = trainSegmenter();

	private Segmenter trainSegmenter() {
		final FeatureSettings featureSettings = new FeatureSettings(GlobalSettings.default3d().build(),
			GroupedFeatures.gauss(),
			GroupedFeatures.differenceOfGaussians(),
			GroupedFeatures.hessian(),
			GroupedFeatures.gradient());
		return Parallelization.runSingleThreaded(() -> Trainer.train(context, image, labelRegions,
			featureSettings));
	}

	@Benchmark
	public Object benchmarkSegment() {
		return Parallelization.runSingleThreaded(() -> segmenter.segment(image));
	}

	@Benchmark
	public Segmenter benchmarkTrain() {
		return trainSegmenter();
	}

	public static void main(String... args) throws RunnerException {
		runBenchmark();
	}

	private static void runForever() {
		StopWatch stopWatch = StopWatch.createAndStart();
		SegmentationBenchmark segmentationBenchmark = new SegmentationBenchmark();
		while (true) {
			System.out.println("start " + stopWatch);
			segmentationBenchmark.benchmarkSegment();
		}
	}

	private static void runBenchmark() throws RunnerException {
		Options options = new OptionsBuilder().include(SegmentationBenchmark.class.getSimpleName())
			.build();
		new Runner(options).run();
	}

	// -- Helper --

	private LabelRegions asLabelRegions(RandomAccessibleInterval<? extends RealType<?>> labeling) {
		Img<UnsignedByteType> ints = ArrayImgs.unsignedBytes(Intervals.dimensionsAsLongArray(labeling));
		RandomAccessibleInterval<LabelingType<String>> labelingTypes = new ImgLabeling<>(ints);
		LoopBuilder.setImages(labeling, labelingTypes).multiThreaded().forEachPixel((i, o) -> {
			if (i.getRealFloat() != 0) o.add(i.toString());
		});
		return new LabelRegions<>(labelingTypes);
	}

}
