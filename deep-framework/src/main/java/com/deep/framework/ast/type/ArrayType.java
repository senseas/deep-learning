package com.deep.framework.ast.type;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.ArrayAccessExpression;
import com.deep.framework.ast.expression.ArrayExpression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.lexer.TokenType;

import java.util.Objects;

public class ArrayType extends Type {

    public ArrayType(Node prarent, Name name) {
        super(name);
        setPrarent(prarent);
    }

    public static void parser(Node node) {
        ArrayAccessExpression.parser(node);
        Stream.of(node.getChildrens()).reduce2((list, a, b) -> {
            if (a instanceof Name && b instanceof ArrayExpression) {
                ArrayType expression = new ArrayType(node, (Name) a);
                list.replace(a, expression);
                list.remove(b);
            } else if (Objects.nonNull(b) && b.equals(TokenType.ELLIPSIS)) {
                ArrayType expression = new ArrayType(node, (Name) a);
                list.replace(a, expression);
                list.remove(b);
            }
        });
    }

    @Override
    public String toString() {
        if (Objects.isNull(getName())) return "[]";
        if (!getChildrens().isEmpty()) return getName().toString().concat("[]");
        return getName().toString().concat("[]");
    }

}