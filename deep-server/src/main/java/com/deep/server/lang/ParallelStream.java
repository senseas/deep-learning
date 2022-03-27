package com.deep.server.lang;

import com.deep.server.framework.DeepClient;
import io.netty.channel.ChannelHandlerContext;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.stream.IntStream;

public class ParallelStream<M> {
    public static List<ChannelHandlerContext> contextList = new ArrayList();

    //而All-reduce需要每个进程都有同样的结果
    void AllReduce(M[] data, BinaryOperator<M> a) {
        M r = Reduce(data, a);
        IntStream.range(0, data.length).forEach(i -> data[i] = r);
        //Scatter(data, (M[]) new Object[]{r});
    }

    //多个进程中的数据按照指定的映射函数进行运算得到最后的结果存在一个进程中
    M Reduce(M[] data, BinaryOperator<M> a) {
        return IntStream.range(0, data.length).mapToObj(i -> data[i]).reduce(a).get();
    }

    //多个进程的数据拼凑在一起
    M[] AllGather(M[] A) {
        return (M[]) IntStream.range(0, A.length).mapToObj(i -> A[i]).toArray();
    }

    //同一份数据分发广播给所有人
    void Broadcast(Object data, int count, Dtype datatype, int root) {
        ChannelHandlerContext context = contextList.get(root);
        Missage missage = new Missage()
                .setData(data)
                .setCount(1)
                .setDatatype(Dtype.INT);
        context.writeAndFlush(missage);
    }

    //可以将不同数据分发给不同的进程
    void Scatter(Object sendData,
                 int sendCount,
                 Dtype sendDtype,
                 Object[] recvData,
                 int recvCount,
                 Dtype recvDtype,
                 int root,
                 Object communicator) {

        //IntStream.range(0, A.length).forEach(i -> B[i] = A[i]);
    }

    public static void main(String[] args) throws InterruptedException {
        IntStream.range(6666, 6667).forEach(port -> {
            new Thread(() -> {
                try {
                    DeepClient.run(port);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();
        });

        Thread.sleep(5000);
        ParallelStream<Integer> rolls = new ParallelStream<Integer>();
        Integer[] data = new Integer[100];
        rolls.Broadcast(1, 1, Dtype.INT, 0);
        System.out.println(data);
    }
}