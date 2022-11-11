package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.type.Type;

import static com.deep.framework.ast.lexer.TokenType.*;

public class VariableDeclarator extends Node {

    private Name name;

    private Expression initializer;

    private Type type;

    public void parser(Node node) {
        for (Object o : node.getChildrens()) {
            if (o.equals(INT)) {

            } else if (o.equals(LONG)) {

            } else if (o.equals(FLOAT)) {

            } else if (o.equals(DOUBLE)) {

            } else if (o.equals(BYTE)) {

            } else if (o.equals(CHAR)) {

            } else if (o.equals(BOOLEAN)) {

            }
        }
    }

}
