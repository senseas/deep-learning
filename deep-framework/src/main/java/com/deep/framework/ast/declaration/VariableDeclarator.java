package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.type.Type;

import static com.deep.framework.ast.lexer.TokenType.*;

public class VariableDeclarator extends Node {
    private Type type;
    private Name name;
    private Expression initializer;

    public static void parser(Node node) {

        Stream.of(node.getChildrens()).reduce((c, a, b) -> {
            if (a.equals(INT)) {
                return;
            } else if (a.equals(LONG)) {
                return;
            } else if (a.equals(FLOAT)) {
                return;
            } else if (a.equals(DOUBLE)) {
                return;
            } else if (a.equals(BYTE)) {
                return;
            } else if (a.equals(CHAR)) {
                return;
            } else if (a.equals(BOOLEAN)) {
                return;
            }
        });
    }

}
