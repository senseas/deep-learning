package com.deep.server.framework;

import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;

public class DeepServer {

    public static void run(int port) throws Exception {
        //创建两个线程组 boosGroup、workerGroup
        EventLoopGroup bossGroup = new NioEventLoopGroup();
        EventLoopGroup workerGroup = new NioEventLoopGroup();
        try {
            //创建服务端的启动对象，设置参数
            ServerBootstrap bootstrap = new ServerBootstrap();
            //设置两个线程组boosGroup和workerGroup
            bootstrap.group(bossGroup, workerGroup)
            //设置服务端通道实现类型
            .channel(NioServerSocketChannel.class)
            //设置线程队列得到连接个数
            .option(ChannelOption.SO_BACKLOG, 128)
            //设置保持活动连接状态
            .childOption(ChannelOption.SO_KEEPALIVE, true)
            //使用匿名内部类的形式初始化通道对象
            .childHandler(new ChannelInitializer<SocketChannel>() {
                @Override
                protected void initChannel(SocketChannel channel) {
                    //给pipeline管道设置处理器
                    channel.pipeline().addLast(new ProtostuffEncoder());
                    channel.pipeline().addLast(new ProtostuffDecoder());
                    channel.pipeline().addLast(new ProtoStuffServerHandler());
                }
            })//给workerGroup的EventLoop对应的管道设置处理器
            //绑定端口号，启动服务端
            .bind(port).sync()
            //对关闭通道进行监听
            .channel().closeFuture().sync();
        } finally {
            bossGroup.shutdownGracefully();
            workerGroup.shutdownGracefully();
        }
    }

}