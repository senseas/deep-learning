package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import static com.deep.framework.ast.lexer.TokenType.SWITCH;

@Data
public class SwitchStatement extends Statement {
    private BlockStatement body;
    private ParametersExpression parameters;
    private static SwitchStatement statement;

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.equals(SWITCH) && n instanceof ParametersExpression) {
                    //create SwitchNode and set Prarent，Parameters
                    statement = new SwitchStatement();
                    statement.setPrarent(node);
                    statement.setParameters((ParametersExpression) n);

                    SwitchEntry.parser(b);
                    b.setPrarent(statement);
                    statement.setBody((BlockStatement) b);
                    statement.getChildrens().addAll(n, b);

                    //remove SwitchNode and Parameters
                    a.getChildrens().removeAll(m, n);
                    node.replace(a, statement);
                    node.getChildrens().remove(b);
                    list.remove(b);
                }
            });
        });
    }

}