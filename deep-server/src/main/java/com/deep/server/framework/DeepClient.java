package com.deep.server.framework;

import io.netty.bootstrap.Bootstrap;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;

public class DeepClient {

    public static void run(int port) throws Exception {
        NioEventLoopGroup eventExecutors = new NioEventLoopGroup();
        try {
            //创建bootstrap对象，配置参数
            Bootstrap bootstrap = new Bootstrap();
            //设置线程组
            bootstrap.group(eventExecutors)
            //设置客户端的通道实现类型
            .channel(NioSocketChannel.class)
            //使用匿名内部类初始化通道
            .handler(new ChannelInitializer<SocketChannel>() {
                @Override
                protected void initChannel(SocketChannel channel) {
                    //添加客户端通道的处理器
                    channel.pipeline().addLast(new ProtostuffEncoder());
                    channel.pipeline().addLast(new ProtostuffDecoder());
                    channel.pipeline().addLast(new DeepClientHandler());
                }
            })
            //连接服务端
            .connect("127.0.0.1", port).sync()
            //对通道关闭进行监听
            .channel().closeFuture().sync();
        } finally {
            //关闭线程组
            eventExecutors.shutdownGracefully();
        }
    }
}