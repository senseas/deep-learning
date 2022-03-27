package com.deep.server.framework;

import com.deep.server.lang.Missage;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class ProtoStuffServerHandler extends ChannelInboundHandlerAdapter {

    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) {
        Missage person = (Missage) msg;
        log.info("Missage:{}", person);
    }

}