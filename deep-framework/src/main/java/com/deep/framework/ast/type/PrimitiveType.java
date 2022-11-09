package com.deep.framework.ast.type;

import com.deep.framework.ast.expression.Name;

public class PrimitiveType extends Type {

    private Type type;

    public PrimitiveType(Name name) {
        super(name);
    }

}
