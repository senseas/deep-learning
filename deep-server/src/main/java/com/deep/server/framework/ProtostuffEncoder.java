package com.deep.server.framework;

import com.deep.server.lang.Missage;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToByteEncoder;
import io.protostuff.LinkedBuffer;
import io.protostuff.ProtobufIOUtil;
import io.protostuff.Schema;
import io.protostuff.runtime.RuntimeSchema;

public class ProtostuffEncoder extends MessageToByteEncoder<Missage> {

    @Override
    protected void encode(ChannelHandlerContext ctx, Missage msg, ByteBuf out) {
        LinkedBuffer buffer = LinkedBuffer.allocate(1024);
        Schema<Missage> schema = RuntimeSchema.getSchema(Missage.class);
        byte[] array = ProtobufIOUtil.toByteArray(msg, schema, buffer);
        out.writeBytes(array);
    }

}