package com.deep.framework.cuda;

import com.deep.framework.cublas.CublasConfig;
import com.deep.framework.cudnn.CudnnConfig;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Tenserx;
import jcuda.jcublas.cublasHandle;
import jcuda.jcudnn.cudnnHandle;
import jcuda.runtime.cudaStream_t;
import lombok.Data;

import java.io.Serializable;
import java.util.Objects;

import static com.deep.framework.cuda.Cuda.copyDataDeviceToHost;
import static com.deep.framework.cuda.Cuda.copyDataHostToDevice;
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
        cudnnHandle handle = CudnnConfig.getCudnnHandle(deviceId);
        cudnnSetStream(handle, stream);
        return handle;
    }

    public Tenserx getDeviceData(Tensor tensor) {
        Tenserx deviceData = tensor.getDeviceDataMap().get(deviceId);
        if (Objects.isNull(deviceData)) {
            tensor.getDeviceDataMap().put(deviceId, deviceData = new Tenserx(tensor.getData(), tensor.getShape(), tensor.getOffset(), deviceId));
        } else {
            copyDataHostToDevice(tensor.getData(), deviceData.deviceData, tensor.getOffset(), tensor.size(), stream);
        }
        return deviceData;
    }

    public Tenserx getDeviceGrad(Tensor tensor) {
        Tenserx deviceGrad = tensor.getDeviceGradMap().get(deviceId);
        if (Objects.isNull(deviceGrad)) {
            tensor.getDeviceGradMap().put(deviceId, deviceGrad = new Tenserx(tensor.getGrad(), tensor.getShape(), tensor.getOffset(), deviceId));
        } else {
            copyDataHostToDevice(tensor.getGrad(), deviceGrad.deviceData, tensor.getOffset(), tensor.size(), stream);
        }
        return deviceGrad;
    }

    public void copyDataToHost(Tensor tensor) {
        Tenserx deviceData = tensor.getDeviceDataMap().get(deviceId);
        if (Objects.isNull(deviceData)) return;
        copyDataDeviceToHost(tensor.getData(), deviceData.deviceData, tensor.getOffset(), tensor.size(), stream);
    }

    public void copyGradToHost(Tensor tensor) {
        Tenserx deviceGrad = tensor.getDeviceGradMap().get(deviceId);
        if (Objects.isNull(deviceGrad)) return;
        copyDataDeviceToHost(tensor.getGrad(), deviceGrad.deviceData, tensor.getOffset(), tensor.size(), stream);
    }

    public void clear() {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

}