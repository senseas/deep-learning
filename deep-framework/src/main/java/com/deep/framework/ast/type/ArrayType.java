package com.deep.framework.ast.type;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.ArrayAccessExpression;
import com.deep.framework.ast.expression.ArrayExpression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.lexer.TokenType;

import java.util.Objects;

public class ArrayType extends Type {
    private Type type;

    public ArrayType(Name name) {
        super(name);
    }

    public ArrayType(Type type) {
        super(type.getName());
        this.type = type;
    }

    public static void parser(Node node) {
        ArrayAccessExpression.parser(node);
        Stream.of(node.getChildrens()).reduce2((list, a, b) -> {
            if (a instanceof Name && b instanceof ArrayExpression) {
                ArrayType expression = new ArrayType((Name) a);
                list.replace(a, expression);
                list.remove(b);
            } else if (a instanceof Type && b instanceof ArrayExpression) {
                ArrayType expression = new ArrayType((Type) a);
                list.replace(a, expression);
                list.remove(b);
                list.previous();
            } else if (Objects.nonNull(b) && b.equals(TokenType.ELLIPSIS)) {
                ArrayType expression = new ArrayType((Name) a);
                list.replace(a, expression);
                list.remove(b);
            }
        });
    }

    @Override
    public String toString() {
        if (Objects.isNull(getName())) return "[]";
        if (Objects.nonNull(type)) return type.toString().concat("[]");
        return getName().toString().concat("[]");
    }

}