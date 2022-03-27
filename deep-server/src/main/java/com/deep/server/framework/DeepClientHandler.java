package com.deep.server.framework;

import com.deep.server.lang.ParallelStream;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import io.netty.util.CharsetUtil;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class DeepClientHandler extends ChannelInboundHandlerAdapter {

    public void channelRegistered(ChannelHandlerContext ctx) {
        ctx.fireChannelRegistered();
        ParallelStream.contextList.add(ctx);
    }

    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) {
        //接收服务端发送过来的消息
        ByteBuf byteBuf = (ByteBuf) msg;
        log.info("Response:{}", byteBuf.toString(CharsetUtil.UTF_8));
    }

}