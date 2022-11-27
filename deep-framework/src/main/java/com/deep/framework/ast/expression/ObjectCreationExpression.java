package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.declaration.CallableDeclaration;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import lombok.Data;

import java.util.List;

@Data
public class ObjectCreationExpression extends Expression {
    private Name name;
    private Expression parameters;
    private BlockStatement body;
    private static ObjectCreationExpression expression;

    public ObjectCreationExpression(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.equals(TokenType.NEW) && n instanceof ParametersExpression) {
                    expression = new ObjectCreationExpression(node);
                    expression.setParameters((ParametersExpression) n);
                    expression.setName((Name) n);
                    node.replace(a, expression);

                    if (b instanceof BlockStatement) {
                        expression.setBody((BlockStatement) b);
                        expression.getChildrens().addAll(List.of(m, n, b));
                        node.getChildrens().removeAll(List.of(m, n));
                        list.remove(b);
                    }
                } else if (m.equals(TokenType.NEW) && n instanceof CallableDeclaration o) {
                    expression = new ObjectCreationExpression(node);
                    expression.setParameters(o.getParameters());
                    expression.setName(o.getName());
                    a.replace(m, expression);
                    a.getChildrens().remove(n);

                    if (b instanceof BlockStatement) {
                        expression.setBody((BlockStatement) b);
                        expression.getChildrens().addAll(List.of(m, n, b));
                        node.replaceAndRemove(a, expression, b);
                        list.remove(b);
                    }
                }
            });
        });
    }

    @Override
    public String toString() {
        return "new ".concat(name.toString()).concat("(").concat(parameters.toString()).concat(")");
    }
}