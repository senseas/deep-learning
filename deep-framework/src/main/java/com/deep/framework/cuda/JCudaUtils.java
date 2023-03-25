package com.deep.framework.cuda;

import jcuda.CudaException;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUresult;
import lombok.SneakyThrows;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.logging.Logger;

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
import static jcuda.driver.JCudaDriver.*;


public class JCudaUtils {
    private static final Logger logger = Logger.getLogger(JCudaUtils.class.getName());

    @SneakyThrows
    private static String invokeNvccCubin(String fileName) {
        String outFile = fileName.replace(".cu", ".cubin");
        String command = "nvcc -cubin -dlink -m".concat(System.getProperty("sun.arch.data.model")).concat(" -arch=sm_" + computeComputeCapability()).concat(" ").concat(fileName).concat(" -o ").concat(outFile);
        logger.info("Executing:{}" + command);
        Process process = Runtime.getRuntime().exec(command);
        String errorMessage = new String(toByteArray(process.getErrorStream()));
        String outputMessage = new String(toByteArray(process.getInputStream()));
        int code = process.waitFor();
        if (code != 0) {
            logger.severe("nvcc process exitValue " + code);
            logger.severe("errorMessage:\n" + errorMessage);
            logger.severe("outputMessage:\n" + outputMessage);
            throw new CudaException("Could not create file: " + errorMessage);
        }
        return outFile;
    }

    private static byte[] toByteArray(InputStream inputStream) {
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            baos.write(inputStream.readAllBytes());
            return baos.toByteArray();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    private static int computeComputeCapability() {
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
        int status = cuCtxGetDevice(device);
        if (status != CUresult.CUDA_SUCCESS) {
            throw new CudaException(CUresult.stringFor(status));
        }
        return computeComputeCapability(device);
    }

    private static int computeComputeCapability(CUdevice device) {
        int[] majorArray = {0};
        int[] minorArray = {0};
        cuDeviceGetAttribute(majorArray, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        cuDeviceGetAttribute(minorArray, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        int major = majorArray[0];
        int minor = minorArray[0];
        return major * 10 + minor;
    }

    public static void main(String[] args) {
        invokeNvccCubin("Softmax.cu");
    }
}
