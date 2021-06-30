
package net.imglib2.trainable_segmentation.pixel_feature.filter.structure;

import net.imglib2.trainable_segmentation.gpu.algorithms.GpuEigenvalues;
import net.imglib2.trainable_segmentation.gpu.api.GpuPixelWiseOperation;
import net.imglib2.trainable_segmentation.gpu.algorithms.GpuGauss;
import net.imglib2.trainable_segmentation.gpu.api.GpuImage;
import net.imglib2.trainable_segmentation.gpu.algorithms.GpuNeighborhoodOperation;
import net.imglib2.trainable_segmentation.gpu.api.GpuApi;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.linalg.eigen.EigenValues;
import net.imglib2.trainable_segmentation.gpu.GpuFeatureInput;
import net.imglib2.trainable_segmentation.gpu.api.GpuView;
import net.imglib2.trainable_segmentation.gpu.api.GpuViews;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureOp;
import net.imglib2.trainable_segmentation.utils.views.FastViews;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import org.scijava.plugin.Plugin;
import net.imglib2.algorithm.convolution.Convolution;
import net.imglib2.algorithm.convolution.kernel.Kernel1D;
import net.imglib2.algorithm.convolution.kernel.SeparableKernelConvolution;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.trainable_segmentation.RevampUtils;
import net.imglib2.trainable_segmentation.pixel_feature.filter.AbstractFeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureInput;
import net.imglib2.trainable_segmentation.pixel_feature.filter.hessian.EigenValuesSymmetric3D;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import net.imglib2.view.composite.Composite;
import org.scijava.plugin.Parameter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.StringJoiner;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

@Plugin(type = FeatureOp.class, label = "structure tensor eigenvalues")
public class SingleStructureTensorEigenvaluesFeature extends AbstractFeatureOp {

	@Parameter
	double sigma = 1.0;

	@Parameter
	double integrationScale = 1.0;

	@Override
	public int count() {
		return globalSettings().numDimensions();
	}

	@Override
	public List<String> attributeLabels() {
		List<String> prefix = getPrefix();
		return prefix.stream().map(s -> "structure tensor - " + s + " eigenvalue sigma=" + sigma +
			" integrationScale=" + integrationScale).collect(Collectors.toList());
	}

	@Override
	public void apply(FeatureInput input, List<RandomAccessibleInterval<FloatType>> output) {
		final Interval targetInterval = output.get(0);
		Convolution<NumericType<?>> convolution = gaussConvolution();
		final Interval derivativeInterval = convolution.requiredSourceInterval(targetInterval);
		RandomAccessibleInterval<DoubleType> derivatives = derivatives(input, derivativeInterval);
		RandomAccessibleInterval<DoubleType> products = products(derivatives);
		RandomAccessibleInterval<DoubleType> blurredProducts = Views.interval(products,
			Intervals.addDimension(targetInterval, products.min(products.numDimensions() - 1), products
				.max(products.numDimensions() - 1)));
		convolution.process(products, blurredProducts);
		EigenValues<DoubleType, FloatType> eigenvalueComputer = globalSettings().numDimensions() == 3
			? new EigenValuesSymmetric3D() : EigenValues.symmetric2D();
		LoopBuilder.setImages(FastViews.collapse(blurredProducts), RevampUtils.vectorizeStack(output))
			.forEachPixel(eigenvalueComputer::compute);
	}

	private Convolution<NumericType<?>> gaussConvolution() {
		final Kernel1D[] gauss = globalSettings().pixelSize().stream()
			.map(pixelSize -> gaussKernel(integrationScale / pixelSize))
			.toArray(Kernel1D[]::new);
		return SeparableKernelConvolution.convolution(gauss);
	}

	private Kernel1D gaussKernel(double v) {
		return Kernel1D.symmetric(Gauss3.halfkernels(new double[] { v })[0]);
	}

	private RandomAccessibleInterval<DoubleType> products(
		RandomAccessibleInterval<DoubleType> derivatives)
	{
		Interval interval = RevampUtils.removeLastDimension(derivatives);
		Interval outputInterval = Intervals.addDimension(interval, 0, getNumberOfProducts() - 1);
		RandomAccessibleInterval<DoubleType> output = RevampUtils.createImage(outputInterval,
			new DoubleType());
		LoopBuilder.setImages(FastViews.collapse(derivatives), FastViews.collapse(output)).forEachPixel(
			getProductPerPixelAction());
		return output;
	}

	private RandomAccessibleInterval<DoubleType> derivatives(FeatureInput input,
		Interval derivativeInterval)
	{
		RandomAccessibleInterval<DoubleType> gauss = RevampUtils.createImage(
			Intervals.expand(derivativeInterval, 1), new DoubleType());
		List<Double> pixelSize = globalSettings().pixelSize();
		double[] sigmas = pixelSize.stream().mapToDouble(p -> sigma / p).toArray();
		RandomAccessible<FloatType> original = input.original();
		Gauss3.gauss(sigmas, original, gauss);
		int n = derivativeInterval.numDimensions();
		RandomAccessibleInterval<DoubleType> tmp = RevampUtils.createImage(RevampUtils
			.appendDimensionToInterval(
				derivativeInterval, 0, n - 1), new DoubleType());
		for (int i = 0; i < n; i++)
			derive(gauss, Views.hyperSlice(tmp, n, i), i, pixelSize.get(i));
		return tmp;
	}

	private void derive(RandomAccessible<? extends RealType<?>> input,
		RandomAccessibleInterval<? extends RealType<?>> tmp, int d, double pixelSize)
	{
		final RandomAccessibleInterval<? extends RealType<?>> back = Views.interval(input, Intervals
			.translate(tmp, -1, d));
		final RandomAccessibleInterval<? extends RealType<?>> front = Views.interval(input, Intervals
			.translate(tmp, 1, d));
		final double factor = 0.5 / pixelSize;
		LoopBuilder.setImages(tmp, back, front).forEachPixel((r, b, f) -> {
			r.setReal((f.getRealDouble() - b.getRealDouble()) * factor);
		});
	}

	// -- dimension specific helper methods --

	private List<String> getPrefix() {
		return globalSettings().numDimensions() == 3 ? Arrays.asList("largest", "middle", "smallest")
			: Arrays.asList("largest", "smallest");
	}

	private int getNumberOfProducts() {
		return globalSettings().numDimensions() == 3 ? 6 : 3;
	}

	private BiConsumer<Composite<DoubleType>, Composite<DoubleType>>
		getProductPerPixelAction()
	{
		return globalSettings().numDimensions() == 3
			? SingleStructureTensorEigenvaluesFeature::productPerPixel3d
			: SingleStructureTensorEigenvaluesFeature::productPerPixel2d;
	}

	private static void productPerPixel3d(Composite<DoubleType> i, Composite<DoubleType> o) {
		double x = i.get(0).getRealDouble(), y = i.get(1).getRealDouble(), z = i.get(2).getRealDouble();
		o.get(0).setReal(x * x);
		o.get(1).setReal(x * y);
		o.get(2).setReal(x * z);
		o.get(3).setReal(y * y);
		o.get(4).setReal(y * z);
		o.get(5).setReal(z * z);
	}

	private static void productPerPixel2d(Composite<DoubleType> i, Composite<DoubleType> o) {
		double x = i.get(0).getRealDouble(), y = i.get(1).getRealDouble();
		o.get(0).setReal(x * x);
		o.get(1).setReal(x * y);
		o.get(2).setReal(y * y);
	}

	// -- CLIJ implementation --

	@Override
	public void prefetch(GpuFeatureInput input) {
		double[] integrationSigma = globalSettings().pixelSize().stream().mapToDouble(
			p -> integrationScale / p).toArray();
		double[] gaussSigma = globalSettings().pixelSize().stream().mapToDouble(p -> sigma / p)
			.toArray();
		long[] border = DoubleStream.of(integrationSigma).mapToLong(sigma -> (long) (4 * sigma))
			.toArray();
		Interval derivativeInterval = Intervals.expand(input.targetInterval(), border);
		for (int d = 0; d < globalSettings().numDimensions(); d++)
			input.prefetchDerivative(gaussSigma[d], d, derivativeInterval);
	}

	@Override
	public void apply(GpuFeatureInput input, List<GpuView> output) {
		try (GpuApi scope = input.gpuApi().subScope()) {
			double[] integrationSigma = globalSettings().pixelSize().stream().mapToDouble(
				p -> integrationScale / p).toArray();
			GpuNeighborhoodOperation integrationGauss = GpuGauss.gauss(scope, integrationSigma);
			Interval border = integrationGauss.getRequiredInputInterval(input.targetInterval());
			List<GpuView> derivatives = derivatives(input, border);
			GpuImage products = products(scope, derivatives);
			GpuImage blurredProducts = blur(scope, GpuViews.channels(products), integrationGauss, input
				.targetInterval());
			GpuEigenvalues.symmetric(scope, GpuViews.channels(blurredProducts), output);
		}
	}

	private List<GpuView> derivatives(GpuFeatureInput input, Interval derivativeInterval) {
		double[] gaussSigma = globalSettings().pixelSize().stream().mapToDouble(p -> sigma / p)
			.toArray();
		int n = globalSettings().numDimensions();
		List<GpuView> derivatives = new ArrayList<>(3);
		for (int d = 0; d < n; d++)
			derivatives.add(input.derivative(gaussSigma[d], d, derivativeInterval));
		return derivatives;
	}

	private GpuImage products(GpuApi gpu, List<GpuView> derivatives) {
		int n = derivatives.size();
		int numProducts = n * (n + 1) / 2;
		long[] dimensions = Intervals.dimensionsAsLongArray(derivatives.get(0).dimensions());
		GpuPixelWiseOperation loopBuilder = GpuPixelWiseOperation.gpu(gpu);
		StringJoiner operation = new StringJoiner("; ");
		for (int i = 0; i < derivatives.size(); i++)
			loopBuilder.addInput("derivative" + i, derivatives.get(i));
		GpuImage products = gpu.create(dimensions, numProducts, NativeTypeEnum.Float);
		Iterator<GpuView> iterator = GpuViews.channels(products).iterator();
		for (int i = 0; i < derivatives.size(); i++)
			for (int j = i; j < derivatives.size(); j++) {
				loopBuilder.addOutput("product" + i + j, iterator.next());
				operation.add("product" + i + j + " = derivative" + i + " * derivative" + j);
			}
		loopBuilder.forEachPixel(operation.toString());
		return products;
	}

	private GpuImage blur(GpuApi gpu, List<GpuView> products,
		GpuNeighborhoodOperation integrationGauss, Interval targertInteval)
	{
		long[] dimensions = Intervals.dimensionsAsLongArray(targertInteval);
		GpuImage blurred = gpu.create(dimensions, products.size(), NativeTypeEnum.Float);
		List<GpuView> blurredChannels = GpuViews.channels(blurred);
		for (int i = 0; i < products.size(); i++) {
			integrationGauss.apply(products.get(i), blurredChannels.get(i));
		}
		return blurred;
	}
}
