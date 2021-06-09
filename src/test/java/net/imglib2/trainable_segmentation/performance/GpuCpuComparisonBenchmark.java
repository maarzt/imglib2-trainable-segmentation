
package net.imglib2.trainable_segmentation.performance;

import net.imglib2.img.Img;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.util.Intervals;
import net.imglib2.loops.LoopBuilder;

import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.assertTrue;

public class GpuCpuComparisonBenchmark {

	public static void main(String... args) {
		ParallelSegmentationTask task = new ParallelSegmentationTask();
		// Gpu
		task.setUseGpu(true);
		task.run();
		task.printTimes();
		Img<UnsignedShortType> gpuResult = task.getSegmenation();
		long nanosecondsGpu = task.measuredTime().nanoTime();

		// Cpu
		task.setUseGpu(false);
		task.run();
		task.printTimes();
		Img<UnsignedShortType> cpuResult = task.getSegmenation();
		long nanosecondsCpu = task.measuredTime().nanoTime();

		System.out.println("Speed up: " + (float) nanosecondsCpu / nanosecondsGpu);
		long differences = countDifferentPixels(gpuResult, cpuResult);
		assertTrue(differences < Intervals.numElements(cpuResult) / 10000);
	}

	public static long countDifferentPixels(Img<UnsignedShortType> gpuResult,
		Img<UnsignedShortType> cpuResult)
	{
		AtomicLong counter = new AtomicLong(0);
		LoopBuilder.setImages(gpuResult, cpuResult).forEachPixel((a, b) -> {
			if (!a.equals(b)) counter.incrementAndGet();
		});
		return counter.get();
	}
}
