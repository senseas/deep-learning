package com.deep.server.framework;

import com.deep.server.lang.Missage;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.ByteToMessageDecoder;
import io.protostuff.ProtobufIOUtil;
import io.protostuff.Schema;
import io.protostuff.runtime.RuntimeSchema;

import java.util.List;

public class ProtostuffDecoder extends ByteToMessageDecoder {

    @Override
    protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) {
        Schema<Missage> schema = RuntimeSchema.getSchema(Missage.class);
        Missage person = schema.newMessage();
        byte[] array = new byte[in.readableBytes()];
        in.readBytes(array);
        ProtobufIOUtil.mergeFrom(array, person, schema);
        out.add(person);
    }

}