
package net.imglib2.trainable_segmentation.classification;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;
import net.imglib2.converter.RealTypeConverters;
import net.imglib2.trainable_segmentation.gpu.api.GpuImage;
import net.imglib2.trainable_segmentation.gpu.api.GpuApi;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.imglib2.*;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.trainable_segmentation.gpu.api.GpuCopy;
import net.imglib2.trainable_segmentation.gpu.api.GpuPool;
import net.imglib2.trainable_segmentation.gpu.random_forest.RandomForestPrediction;
import net.imglib2.trainable_segmentation.pixel_feature.calculator.FeatureCalculator;
import net.imglib2.trainable_segmentation.pixel_feature.settings.FeatureSettings;
import net.imglib2.trainable_segmentation.RevampUtils;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import net.imglib2.view.composite.Composite;
import org.scijava.Context;
import preview.net.imglib2.loops.LoopBuilder;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author Matthias Arzt
 */
public class Segmenter {

	private final FeatureCalculator features;

	private final List<String> classNames;

	public Classifier getClassifier() {
		return classifier;
	}

	private weka.classifiers.Classifier classifier;

	private boolean useGpu = false;

	private long[] fixedOutputSize;

	private Segmenter(List<String> classNames, FeatureCalculator features,
		Classifier classifier)
	{
		this.classNames = Collections.unmodifiableList(classNames);
		this.features = Objects.requireNonNull(features);
		this.classifier = Objects.requireNonNull(classifier);
	}

	public Segmenter(Context context, List<String> classNames, FeatureSettings features,
		Classifier classifier)
	{
		this(classNames, new FeatureCalculator(context, features), classifier);
	}

	public void setUseGpu(boolean useGpu) {
		this.useGpu = useGpu;
		features.setUseGpu(useGpu);
	}

	public void setFixedOutputSize(long... cellSize) {
		this.fixedOutputSize = cellSize;
	}

	public FeatureCalculator features() {
		return features;
	}

	public FeatureSettings settings() {
		return features.settings();
	}

	public RandomAccessibleInterval<UnsignedByteType> segment(RandomAccessibleInterval<?> image) {
		return segment(image, new UnsignedByteType());
	}

	private <T extends NativeType<T>> RandomAccessibleInterval<T> createImage(T type,
		Interval interval)
	{
		// FIXME: use this in all places where it's useful.
		long[] size = Intervals.dimensionsAsLongArray(interval);
		long[] min = Intervals.minAsLongArray(interval);
		Img<T> img = new ArrayImgFactory<>(type).create(size);
		return Views.translate(img, min);
	}

	public <T extends IntegerType<T> & NativeType<T>> RandomAccessibleInterval<T> segment(
		RandomAccessibleInterval<?> image, T type)
	{
		Interval interval = features.outputIntervalFromInput(image);
		RandomAccessibleInterval<T> rai = createImage(type, interval);
		segment(rai, Views.extendBorder(image));
		return rai;
	}

	public void segment(RandomAccessibleInterval<? extends IntegerType<?>> out,
		RandomAccessible<?> image)
	{
		Objects.requireNonNull(out);
		Objects.requireNonNull(image);
		if (useGpu)
			segmentGpu(image, out);
		else
			segmentCpu(image, out);
	}

	private void segmentCpu(RandomAccessible<?> image,
		RandomAccessibleInterval<? extends IntegerType<?>> out)
	{
		RandomAccessibleInterval<FloatType> featureValues = features.apply(image, out);
		LoopBuilder.setImages(Views.collapseReal(featureValues), out).multiThreaded().forEachChunk(
			chunk -> {
				CompositeInstance compositeInstance = new CompositeInstance(null, attributesAsArray());
				chunk.forEachPixel((input, output) -> {
					compositeInstance.setSource(input);
					RevampUtils.wrapException(() -> output.setInteger((int) classifier.classifyInstance(
						compositeInstance)));
				});
				return null;
			});
	}

	private void segmentGpu(RandomAccessible<?> image,
		RandomAccessibleInterval<? extends IntegerType<?>> out)
	{
		try (GpuApi scope = GpuPool.borrowGpu()) {
			RandomForestPrediction prediction = new RandomForestPrediction(Cast.unchecked(classifier),
				classNames.size(), features.count());
			GpuImage featureStack = features.applyUseGpu(scope, image, ensureOutputSize(out));
			GpuImage segmentationBuffer = prediction.segment(scope, featureStack);
			copyFromTo(scope, segmentationBuffer, out);
		}
	}

	public RandomAccessibleInterval<? extends RealType<?>> predict(
		RandomAccessibleInterval<?> image)
	{
		Objects.requireNonNull(image);
		Interval outputInterval = features.outputIntervalFromInput(image);
		RandomAccessibleInterval<FloatType> result = RevampUtils.createImage(RevampUtils
			.appendDimensionToInterval(
				outputInterval, 0, classNames.size() - 1), new FloatType());
		predict(result, Views.extendBorder(image));
		return result;
	}

	public void predict(RandomAccessibleInterval<? extends RealType<?>> out,
		RandomAccessible<?> image)
	{
		Objects.requireNonNull(out);
		Objects.requireNonNull(image);
		if (useGpu)
			predictGpu(out, image);
		else
			predictCpu(out, image);
	}

	private void predictCpu(RandomAccessibleInterval<? extends RealType<?>> out,
		RandomAccessible<?> image)
	{
		Interval interval = RevampUtils.removeLastDimension(out);
		RandomAccessibleInterval<FloatType> featureValues = features.apply(image, interval);
		LoopBuilder.setImages(Views.collapseReal(featureValues), Views.collapse(out)).multiThreaded()
			.forEachChunk(
				chunk -> {
					CompositeInstance compositeInstance = new CompositeInstance(null, attributesAsArray());
					chunk.forEachPixel((input, output) -> {
						compositeInstance.setSource(input);
						double[] result = RevampUtils.wrapException(() -> classifier.distributionForInstance(
							compositeInstance));
						for (int i = 0, n = result.length; i < n; i++)
							output.get(i).setReal(result[i]);
					});
					return null;
				});
	}

	private void predictGpu(RandomAccessibleInterval<? extends RealType<?>> out,
		RandomAccessible<?> image)
	{
		Interval interval = RevampUtils.removeLastDimension(out);
		RandomForestPrediction prediction = new RandomForestPrediction(Cast.unchecked(classifier),
			classNames.size(), features.count());
		try (GpuApi scope = GpuPool.borrowGpu()) {
			GpuImage featureStack = features.applyUseGpu(scope, image, ensureOutputSize(interval));
			GpuImage distribution = scope.create(featureStack.getDimensions(), features.count(),
				NativeTypeEnum.Float);
			prediction.distribution(scope, featureStack, distribution);
			copyFromTo(scope, featureStack, out);
		}
	}

	public List<String> classNames() {
		return classNames;
	}

	public Training training() {
		return new MyTrainingData();
	}

	public JsonElement toJsonTree() {
		JsonObject json = new JsonObject();
		json.add("features", features.settings().toJson());
		json.add("classNames", new Gson().toJsonTree(classNames));
		json.add("classifier", ClassifierSerialization.wekaToJson(classifier));
		return json;
	}

	public static Segmenter fromJson(Context context, JsonElement json) {
		JsonObject object = json.getAsJsonObject();
		return new Segmenter(
			context,
			new Gson().fromJson(object.get("classNames"), new TypeToken<List<String>>()
			{}.getType()),
			FeatureSettings.fromJson(object.get("features")),
			ClassifierSerialization.jsonToWeka(object.get("classifier")));
	}

	private class MyTrainingData implements Training {

		final Instances instances;

		final int featureCount;

		MyTrainingData() {
			this.instances = new Instances("segment", new ArrayList<>(attributes()), 1);
			this.featureCount = features.count();
			instances.setClassIndex(featureCount);
		}

		@Override
		public void add(Composite<? extends RealType<?>> featureVector, int classIndex) {
			instances.add(RevampUtils.getInstance(featureCount, classIndex, featureVector));
		}

		@Override
		public void train() {
			RevampUtils.wrapException(() -> classifier.buildClassifier(instances));
		}
	}

	// -- Helper methods --

	private Attribute[] attributesAsArray() {
		List<Attribute> attributes = attributes();
		return attributes.toArray(new Attribute[attributes.size()]);
	}

	private List<Attribute> attributes() {
		Stream<Attribute> featureAttributes = features.attributeLabels().stream().map(Attribute::new);
		Stream<Attribute> classAttribute = Stream.of(new Attribute("class", classNames));
		return Stream.concat(featureAttributes, classAttribute).collect(Collectors.toList());
	}

	private FinalInterval ensureOutputSize(Interval outputInterval) {
		if (fixedOutputSize != null) {
			checkOutputSize(outputInterval);
			return FinalInterval.createMinSize(Intervals.minAsLongArray(outputInterval), fixedOutputSize);
		}
		else
			return new FinalInterval(outputInterval);
	}

	private void checkOutputSize(Interval outputInterval) {
		long[] outputSize = Intervals.dimensionsAsLongArray(outputInterval);
		if (fixedOutputSize.length != outputInterval.numDimensions())
			throw new IllegalArgumentException(
				"The dimensionality of the cell size and output image don't match.");
		for (int d = 0; d < fixedOutputSize.length; d++)
			if (outputSize[d] > fixedOutputSize[d])
				throw new IllegalArgumentException("The output size is bigger than the fixed output size.");
	}

	private void copyFromTo(GpuApi scope, GpuImage source,
		RandomAccessibleInterval<? extends RealType<?>> target)
	{
		// NB: Allows source to be bigger than target. Only copies the relevant part.
		if (Arrays.equals(source.getDimensions(), Intervals.dimensionsAsLongArray(target)))
			GpuCopy.copyFromTo(source, target);
		else {
			RandomAccessibleInterval<? extends RealType<?>> result = scope.pullRAI(source);
			RandomAccessibleInterval<? extends RealType<?>> croppedResult = Views.interval(result, Views
				.zeroMin(target));
			RealTypeConverters.copyFromTo(croppedResult, Views.zeroMin(target));
		}
	}

}