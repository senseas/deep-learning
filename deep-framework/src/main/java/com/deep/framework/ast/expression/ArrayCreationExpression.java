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
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.equals(TokenType.NEW) && n instanceof ArrayType o) {
                    ArrayCreationExpression expression = new ArrayCreationExpression(a, o);
                    a.replace(m, expression);
                    a.getChildrens().remove(n);
                }
            });
        });
    }

    @Override
    public String toString() {
        return "new ".concat(type.toString());
    }
}