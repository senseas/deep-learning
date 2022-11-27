package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import java.util.List;

import static com.deep.framework.ast.lexer.TokenType.SYNCHRONIZED;

@Data
public class SynchronizedStatement extends Statement {
    private List<Expression> initialization;
    private Expression compare;
    private List<Expression> update;
    private BlockStatement body;
    private ParametersExpression parameters;
    private static SynchronizedStatement statement;

    public static void parser(Node node) {
        if (node instanceof SynchronizedStatement) return;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.equals(SYNCHRONIZED) && n instanceof ParametersExpression) {
                    //create SynchronizedNode and set Prarent，Parameters
                    statement = new SynchronizedStatement();
                    statement.setPrarent(node);
                    statement.setParameters((ParametersExpression) n);

                    //remove SynchronizedNode and Parameters
                    a.getChildrens().removeAll(List.of(m, n));
                    node.replace(a, statement);

                    if (b instanceof BlockStatement) {
                        b.setPrarent(statement);
                        statement.setBody((BlockStatement) b);
                        statement.getChildrens().addAll(List.of(n, b));
                        node.getChildrens().remove(b);
                        list.remove(b);
                    } else {
                        throw new RuntimeException("SynchronizedStatement parser error ".concat(node.toString()));
                    }
                }
            });
        });
    }

}