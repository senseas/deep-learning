package com.deep.framework;

import com.jogamp.common.nio.Buffers;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLCommandQueue.Mode;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLPlatform;
import com.jogamp.opencl.CLProgram;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingUtilities;

import static com.jogamp.opencl.CLProgram.*;

/**
 * Computes the classical gamma correction for a given image.
 * http://en.wikipedia.org/wiki/Gamma_correction
 * @author Michael Bien
 */
public class CLSimpleGammaCorrection {

    private static InputStream getStreamFor(String filename) {
        return CLSimpleGammaCorrection.class.getResourceAsStream(filename);
    }
    
    public static BufferedImage readImage(String filename) throws IOException {
        return ImageIO.read(getStreamFor(filename));
    }

    private static BufferedImage createImage(int width, int height, CLBuffer<FloatBuffer> buffer) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        float[] pixels = new float[buffer.getBuffer().capacity()];
        buffer.getBuffer().get(pixels).rewind();
        image.getRaster().setPixels(0, 0, width, height, pixels);
        return image;
    }
    
    public static void main(String[] args) throws IOException {
        
        // find a CL implementation
        CLPlatform platform = CLPlatform.getDefault(/*type(CPU)*/);
        
        CLContext context = CLContext.create(platform.getMaxFlopsDevice());
        
        try{
            //load and compile program for the chosen device
            CLProgram program = context.createProgram(new FileInputStream(new File("D:\\GitHub\\deep-learning\\deep-framework\\src\\test\\java\\com\\deep\\framework\\Gamma.cl")));
            program.build(CompilerOptions.FAST_RELAXED_MATH);
            assert program.isExecutable();
            
            // load image
            BufferedImage image =ImageIO.read(new FileInputStream(new File("D:\\GitHub\\deep-learning\\deep-framework\\src\\test\\java\\com\\deep\\framework\\lena.png")));
            assert image.getColorModel().getNumComponents() == 3;
            
            float[] pixels = image.getRaster().getPixels(0, 0, image.getWidth(), image.getHeight(), (float[])null);
            
            // copy to direct float buffer
            FloatBuffer fb = Buffers.newDirectFloatBuffer(pixels);
            
            // allocate a OpenCL buffer using the direct fb as working copy
            CLBuffer<FloatBuffer> buffer = context.createBuffer(fb, CLBuffer.Mem.READ_WRITE);
            
            // creade a command queue with benchmarking flag set
            CLCommandQueue queue = context.getDevices()[0].createCommandQueue(Mode.PROFILING_MODE);
            
            int localWorkSize = queue.getDevice().getMaxWorkGroupSize(); // Local work size dimensions
            int globalWorkSize = roundUp(localWorkSize, fb.capacity());  // rounded up to the nearest multiple of the localWorkSize
            
            // create kernel and set function parameters
            CLKernel kernel = program.createCLKernel("gamma");
            
            // original lenna
            show(image, 0, 50, "reference");
            
            // a few gamma corrected versions
            float gamma = 0.5f;
            gammaCorrection(gamma, queue, kernel, buffer, localWorkSize, globalWorkSize);
            show(createImage(image.getWidth(), image.getHeight(), buffer), image.getWidth()/2, 50, "gamma="+gamma);
            
            gamma = 1.5f;
            gammaCorrection(gamma, queue, kernel, buffer, localWorkSize, globalWorkSize);
            show(createImage(image.getWidth(), image.getHeight(), buffer), image.getWidth()/2*2, 50, "gamma="+gamma);
            
            gamma = 2.0f;
            gammaCorrection(gamma, queue, kernel, buffer, localWorkSize, globalWorkSize);
            show(createImage(image.getWidth(), image.getHeight(), buffer), image.getWidth()/2*3, 50, "gamma="+gamma);
            
        }finally{
            context.release();
        }
        
    }

    private static void gammaCorrection(float gamma, CLCommandQueue queue, CLKernel kernel, CLBuffer<FloatBuffer> buffer, int localWorkSize, int globalWorkSize) {
       
        float scaleFactor = (float) Math.pow(255, 1.0f-gamma);
        
        // setup kernel
        kernel.putArg(buffer).putArg(gamma).putArg(scaleFactor).putArg(buffer.getNIOSize()).rewind();  
        
//        CLEventList list = new CLEventList(1);
        
        queue.putWriteBuffer(buffer, false);                                      // upload image
        queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize/*, list*/);   // execute program
        queue.putReadBuffer(buffer, true);                                        // read results back (blocking read)
        
//        CLEvent event = list.getEvent(0);
//        System.out.println((event.getProfilingInfo(END)
//                          - event.getProfilingInfo(START))/1000000.0);
//        
//        long res = queue.getDevice().getProfilingTimerResolution();
//        System.out.println(res);
        
    }

    private static void show(final BufferedImage image, final int x, final int y, final String title) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                JFrame frame = new JFrame("gamma correction example ["+title+"]");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.add(new JLabel(new ImageIcon(image)));
                frame.pack();
                frame.setLocation(x, y);
                frame.setVisible(true);
            }
        });
    }
    
    private static int roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }
    
}