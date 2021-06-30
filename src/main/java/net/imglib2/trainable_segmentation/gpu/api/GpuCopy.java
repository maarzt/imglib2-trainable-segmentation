
package net.imglib2.trainable_segmentation.gpu.api;

import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.RealTypeConverters;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.basictypeaccess.array.ArrayDataAccess;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.Arrays;

public class GpuCopy {

	public static void copyFromTo(GpuApi gpu, GpuView src, GpuView dst) {
		GpuPixelWiseOperation.gpu(gpu)
			.addInput("s", src)
			.addOutput("d", dst).forEachPixel("d = s");
	}

	public static void copyFromTo(RandomAccessibleInterval<? extends RealType<?>> source,
		GpuImage target)
	{
		checkEqualDimensions(source, target);
		RealType<?> sourceType = Util.getTypeFromInterval(source);
		RealType<?> targetType = getImgLib2Type(target.getNativeType());
		Object array = getBackingArrayOrNull(source);
		if (array != null && sourceType.getClass() == targetType.getClass()) {
			target.clearCLBuffer().readFrom(wrapAsBuffer(array), true);
		}
		else {
			RandomAccessibleInterval<RealType<?>> tmp = new ArrayImgFactory<>((NativeType) targetType)
				.create(source);
			RealTypeConverters.copyFromTo(Views.zeroMin(source), tmp);
			copyFromTo(tmp, target);
		}
	}

	public static void copyFromTo(GpuImage source,
		RandomAccessibleInterval<? extends RealType<?>> target)
	{
		checkEqualDimensions(target, source);
		RealType<?> sourceType = getImgLib2Type(source.getNativeType());
		RealType<?> targetType = Util.getTypeFromInterval(target);
		Object array = getBackingArrayOrNull(target);
		if (array != null && sourceType.getClass() == targetType.getClass()) {
			source.clearCLBuffer().writeTo(wrapAsBuffer(array), true);
		}
		else {
			RandomAccessibleInterval<RealType<?>> tmp = new ArrayImgFactory<>((NativeType) sourceType)
				.create(target);
			copyFromTo(source, tmp);
			RealTypeConverters.copyFromTo(tmp, Views.zeroMin(target));
		}
	}

	private static void checkEqualDimensions(RandomAccessibleInterval<? extends RealType<?>> rai,
		GpuImage gpuImage)
	{
		long[] a = Intervals.dimensionsAsLongArray(rai);
		long[] b = gpuImage.getDimensions();
		if (gpuImage.getNumberOfChannels() == 1 && Arrays.equals(a, b))
			return;
		b = Arrays.copyOf(b, b.length + 1);
		b[b.length - 1] = gpuImage.getNumberOfChannels();
		if (Arrays.equals(a, b))
			return;
		throw new IllegalArgumentException("Dimensions don't match.");
	}

	public static RealType<?> getImgLib2Type(NativeTypeEnum nativeType) {
		switch (nativeType) {
			case Byte:
				return new ByteType();
			case UnsignedByte:
				return new UnsignedByteType();
			case Short:
				return new ShortType();
			case UnsignedShort:
				return new UnsignedShortType();
			case Int:
				return new IntType();
			case UnsignedInt:
				return new UnsignedIntType();
			case Long:
				return new LongType();
			case UnsignedLong:
				return new UnsignedLongType();
			case HalfFloat:
				throw new UnsupportedOperationException();
			case Float:
				return new FloatType();
			case Double:
				return new DoubleType();
		}
		throw new UnsupportedOperationException();
	}

	private static Buffer wrapAsBuffer(Object array) {
		if (array instanceof byte[])
			return ByteBuffer.wrap((byte[]) array);
		if (array instanceof float[])
			return FloatBuffer.wrap((float[]) array);
		if (array instanceof short[])
			return ShortBuffer.wrap((short[]) array);
		if (array instanceof int[])
			return IntBuffer.wrap((int[]) array);
		throw new UnsupportedOperationException();
	}

	private static Object getBackingArrayOrNull(RandomAccessibleInterval<?> image) {
		if (!(image instanceof ArrayImg))
			return null;
		Object access = ((ArrayImg<?, ?>) image).update(null);
		if (!(access instanceof ArrayDataAccess))
			return null;
		return ((ArrayDataAccess) access).getCurrentStorageArray();
	}

	public static NativeTypeEnum getNativeTypeEnum(
		RandomAccessibleInterval<? extends RealType> image)
	{
		RealType type = Util.getTypeFromInterval(image);
		if (type instanceof FloatType)
			return NativeTypeEnum.Float;
		if (type instanceof UnsignedShortType)
			return NativeTypeEnum.UnsignedShort;
		throw new UnsupportedOperationException();
	}
}
