package com.deep.framework.cuda;

import com.deep.framework.cublas.CublasConfig;
import com.deep.framework.cudnn.CudnnConfig;
import com.deep.framework.graph.Tensor;
import jcuda.Pointer;
import jcuda.jcublas.cublasHandle;
import jcuda.jcudnn.cudnnHandle;
import jcuda.runtime.cudaStream_t;
import lombok.Data;

import java.io.Serializable;
import java.util.Objects;

import static com.deep.framework.cuda.Cuda.*;
import static jcuda.jcublas.JCublas2.cublasSetStream;
import static jcuda.jcudnn.JCudnn.cudnnSetStream;
import static jcuda.runtime.JCuda.*;

@Data
public class CudaContext implements Serializable {

    public final int deviceId;
    public final cudaStream_t stream;

    public CudaContext(Tensor output) {
        deviceId = output.getDeviceId();
        cudaSetDevice(deviceId);

        stream = new cudaStream_t();
        cudaStreamCreate(stream);
    }

    public cublasHandle getCublasHandle() {
        cublasHandle handle = CublasConfig.getCublasHandle(deviceId);
        cublasSetStream(handle, stream);
        return handle;
    }

    public cudnnHandle getCudnnHandle() {
        cudnnHandle cudnn = CudnnConfig.getCudnnHandle(deviceId);
        cudnnSetStream(cudnn, stream);
        return cudnn;
    }

    public Pointer getDeviceData(Tensor tensor) {
        jcuda.Pointer deviceData = tensor.getDeviceDataMap().get(deviceId);
        if (Objects.isNull(deviceData)) {
            tensor.getDeviceDataMap().put(deviceId, deviceData = createDevicePointer(tensor.getData(), deviceId));
        } else {
            copyDataHostToDevice(tensor.getData(), deviceData, stream);
        }
        return deviceData;
    }

    public Pointer getDeviceGrad(Tensor tensor) {
        Pointer deviceGrad = tensor.getDeviceGradMap().get(deviceId);
        if (Objects.isNull(deviceGrad)) {
            tensor.getDeviceGradMap().put(deviceId, deviceGrad = createDevicePointer(tensor.getGrad(), deviceId));
        } else {
            copyDataHostToDevice(tensor.getGrad(), deviceGrad, stream);
        }
        return deviceGrad;
    }

    public void copyDataToHost(Tensor tensor) {
        Pointer deviceData = tensor.getDeviceDataMap().get(deviceId);
        if (Objects.isNull(deviceData)) return;
        copyDataDeviceToHost(tensor.getData(), deviceData, stream);
    }

    public void copyGradToHost(Tensor tensor) {
        Pointer deviceGrad = tensor.getDeviceGradMap().get(deviceId);
        if (Objects.isNull(deviceGrad)) return;
        copyDataDeviceToHost(tensor.getGrad(), deviceGrad, stream);
    }

    public void clear() {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

}