package com.deep.framework.ast.type;

import com.deep.framework.ast.expression.Name;

public class ReferenceType extends Type {
    private Name name;

    public ReferenceType(Name name) {
        super(name);
        this.name = name;
    }

}