package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.type.Type;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.INSTANCEOF;

@Data
public class InstanceOfDeclaration extends Declaration {
    private Expression expression;
    private Type type;

    public InstanceOfDeclaration(Node prarent, Expression expression, Type type) {
        super(prarent);
        this.expression = expression;
        this.type = type;

        this.expression.setPrarent(this);
        this.type.setPrarent(this);

        getChildrens().addAll(expression, type);
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((list, a, b, c) -> {
            if (a instanceof Name && Objects.nonNull(b) && b.equals(INSTANCEOF)) {
                InstanceOfDeclaration declare = new InstanceOfDeclaration(node, (Expression) a, (Type) c);
                node.replace(a, declare);
                list.removeAll(List.of(b, c));
            }
        });
    }

    @Override
    public String toString() {
        return expression.toString().concat(INSTANCEOF.toString()).concat(type.toString());
    }
}