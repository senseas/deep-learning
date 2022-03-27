package com.deep.server.lang;

import lombok.Data;
import lombok.experimental.Accessors;

@Data
@Accessors(chain = true)
public class Missage {
    Object data;
    int count;
    Dtype datatype;
}