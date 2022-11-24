package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.statement.BlockStatement;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.ARROW;

@Data
public class LambdaExpression extends Expression {
    public ParametersExpression parameters;
    private BlockStatement body;

    private static LambdaExpression expression;

    public LambdaExpression(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m instanceof ParametersExpression && Objects.nonNull(n) && n.equals(ARROW)) {
                    expression = new LambdaExpression(node);
                    expression.setParameters((ParametersExpression) m);
                    if (Objects.nonNull(b) && b instanceof BlockStatement) {
                        expression.setBody((BlockStatement) b);
                        expression.getChildrens().addAll(List.of(m, b));
                        node.replaceAndRemove(a, expression, b);
                        list.remove(b);
                    } else {
                        BlockStatement block = new BlockStatement(expression);
                        block.setChildrens(a.getChildrens());
                        block.getChildrens().remove(m);
                        block.getChildrens().remove(n);

                        node.replace(a, expression);
                        expression.setBody(block);
                        expression.getChildrens().addAll(List.of(n, block));
                    }
                } else if (m instanceof Name && Objects.nonNull(n) && n.equals(ARROW)) {
                    expression = new LambdaExpression(node);
                    ParametersExpression parameters = new ParametersExpression(expression);
                    parameters.getChildrens().add(m);
                    expression.setParameters(parameters);
                    if (Objects.nonNull(b) && b instanceof BlockStatement) {
                        expression.setBody((BlockStatement) b);
                        expression.getChildrens().addAll(List.of(m, b));
                        node.replaceAndRemove(a, expression, b);
                        list.remove(b);
                    } else {
                        BlockStatement block = new BlockStatement(expression);
                        block.setChildrens(a.getChildrens());
                        block.getChildrens().remove(m);
                        block.getChildrens().remove(n);

                        node.replace(a, expression);
                        expression.setBody(block);
                        expression.getChildrens().addAll(List.of(parameters, block));
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
