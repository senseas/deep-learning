package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.type.ArrayType;
import com.deep.framework.ast.type.Type;
import lombok.Data;

@Data
public class ArrayCreationExpression extends Expression {
    private Type type;

    public ArrayCreationExpression(Node prarent, Type type) {
        super(prarent);
        this.type = type;
        this.type.setPrarent(this);
        getChildrens().add(type);
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            if (a.equals(TokenType.NEW) && b instanceof ArrayType o) {
                ArrayCreationExpression expression = new ArrayCreationExpression(node, o);
                node.replace(a, expression);
                node.getChildrens().remove(b);
            }
        });
    }

    @Override
    public String toString() {
        return "new ".concat(type.toString());
    }
}