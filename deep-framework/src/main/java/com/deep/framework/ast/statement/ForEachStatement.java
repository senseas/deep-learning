package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.FOR;

@Data
public class ForEachStatement extends Statement {
    private List<Expression> initialization;
    private Expression compare;
    private List<Expression> update;
    private BlockStatement body;
    private ParametersExpression parameters;
    private static ForEachStatement statement;

    public static void parser(Node node) {
        if (node instanceof ForStatement) return;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.equals(FOR) && n instanceof ParametersExpression) {
                    //create ForNode and set Prarent，Parameters
                    statement = new ForEachStatement();
                    statement.setPrarent(node);
                    statement.setParameters((ParametersExpression) n);

                    //remove ForNode and Parameters
                    a.getChildrens().removeAll(List.of(m, n));
                    node.replace(a, statement);

                    if (Objects.nonNull(b) && b instanceof BlockStatement) {
                        b.setPrarent(statement);
                        statement.setBody((BlockStatement) b);
                        statement.getChildrens().addAll(List.of(n, b));
                        node.remove(b);
                        list.remove(b);
                    } else {
                        BlockStatement block = new BlockStatement(statement);
                        block.setChildrens(a.getChildrens());
                        statement.setBody(block);
                        statement.getChildrens().addAll(List.of(n, block));
                    }

                }
            });
        });
    }

}