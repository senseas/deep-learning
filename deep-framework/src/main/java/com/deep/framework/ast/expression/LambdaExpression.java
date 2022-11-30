package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.statement.BlockStatement;
import lombok.Data;

import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.ARROW;

@Data
public class LambdaExpression extends Expression {
    private ParametersExpression parameters;
    private BlockStatement body;
    private static LambdaExpression expression;

    public LambdaExpression(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (node instanceof LambdaExpression) return;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m instanceof ParametersExpression && Objects.nonNull(n) && n.equals(ARROW)) {
                    expression = new LambdaExpression(node);
                    expression.setParameters((ParametersExpression) m);
                    expression.getChildrens().add(m);
                    if (b instanceof BlockStatement) {
                        expression.setBody((BlockStatement) b);
                        expression.getChildrens().add(b);
                        node.replaceAndRemove(a, expression, b);
                        list.remove(b);
                    } else {
                        BlockStatement block = new BlockStatement(expression);
                        block.getChildrens().addAll(a.getChildrens());
                        block.getChildrens().removeAll(m, n);

                        expression.setBody(block);
                        expression.getChildrens().add(block);
                        a.getChildrens().remove(n);
                        a.getChildrens().removeAll(block.getChildrens());
                        a.replace(m, expression);
                    }
                } else if (m instanceof Name && Objects.nonNull(n) && n.equals(ARROW)) {
                    expression = new LambdaExpression(node);
                    ParametersExpression parameters = new ParametersExpression(expression);
                    parameters.getChildrens().add(m);
                    expression.setParameters(parameters);
                    expression.getChildrens().add(parameters);
                    if (b instanceof BlockStatement) {
                        expression.setBody((BlockStatement) b);
                        expression.getChildrens().addAll(m, b);
                        node.replaceAndRemove(m, expression, b);
                        list.remove(b);
                    } else {
                        BlockStatement block = new BlockStatement(expression);
                        block.getChildrens().addAll(a.getChildrens());
                        block.getChildrens().removeAll(m, n);

                        expression.setBody(block);
                        expression.getChildrens().add(block);
                        a.getChildrens().remove(n);
                        a.getChildrens().removeAll(block.getChildrens());
                        a.replace(m, expression);
                    }
                }
            });
        });
    }

    @Override
    public String toString() {
        return parameters.toString().concat("->").concat(body.toString());
    }
}