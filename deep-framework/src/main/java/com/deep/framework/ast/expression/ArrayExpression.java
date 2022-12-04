package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import lombok.Data;

@Data
public class ArrayExpression extends Expression {
    public ArrayExpression(Node prarent) {
        super(prarent);
    }
}